import os
import cv2
import time
import torch
import queue
import threading
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import robomail.vision as vis
from frankapy import FrankaArm
from pcl_utils import *
from pointBERT.tools import builder
from imitate_clay_episodes import *
from embeddings.embeddings import EncoderHead
from pointBERT.utils.config import cfg_from_yaml_file
from scipy.spatial.transform import Rotation

'''
This is the generic script for the clay hardware experiments. It will save all the necessary information to
document each experiment. This includes the following:
    - RGB image from each camera
    - goal point cloud
    - state point clouds
    - number of actions to completion
    - real-world time to completion
    - chamfer distance between final state and goal
    - earth mover's distance between final state and goal
    - video from camera 6 recording the entire experimental run
'''

def goto_grasp(fa, x, y, z, rx, ry, rz, d):
	"""
	Parameterize a grasp action by the position [x,y,z] Euler angle rotation [rx,ry,rz], and width [d] of the gripper.
	This function was designed to be used for clay moulding, but in practice can be applied to any task.

	:param fa:  franka robot class instantiation
	"""
	pose = fa.get_pose()
	starting_rot = pose.rotation
	orig = Rotation.from_matrix(starting_rot)
	orig_euler = orig.as_euler('xyz', degrees=True)
	rot_vec = np.array([rx, ry, rz])
	new_euler = orig_euler + rot_vec
	r = Rotation.from_euler('xyz', new_euler, degrees=True)
	pose.rotation = r.as_matrix()
	pose.translation = np.array([x, y, z])

	fa.goto_pose(pose)
	fa.goto_gripper(d, force=60.0)
	time.sleep(3)

def experiment_loop(fa, cam2, cam3, cam4, cam5, pcl_vis, save_path, goal_str, ckpt_dir, done_queue, centered_action):
    '''
    '''
    # load in policy config
    with open(ckpt_dir + '/policy_config.json') as json_file:
        policy_config = json.load(json_file)
    policy_class = 'ACT'

    # load in the modifiers from params.json based on ckpt_dir
    with open(ckpt_dir + '/params.json') as json_file:
        params_config = json.load(json_file)

    # initialize the models
    device = torch.device('cuda')
    testconfig = cfg_from_yaml_file('pointBERT/cfgs/PointTransformer.yaml')
    testmodel_config = testconfig.model
    pointbert = builder.model_builder(testmodel_config)
    testweights_path = ckpt_dir + '/pointbert_statedict.zip' 
    pointbert.load_state_dict(torch.load(testweights_path))
    pointbert.to(device)

    # load the projection head from ckpt
    enc_checkpoint = torch.load(ckpt_dir + '/encoder_best_checkpoint.zip', map_location=torch.device('cpu'))  # '/encoder_best_checkpoint.zip'
    projection_head = enc_checkpoint['encoder_head'].to(device)

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')

    # fixed parameters
    temporal_agg = False
    state_dim = 5
    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']
    with open(ckpt_dir + '/params.json') as json_file:
        params_config = json.load(json_file)
    concat_goal = params_config['concat_goal']
    delta_goal = params_config['delta_goal']
    no_pos_embed = params_config['no_pos_embed']

    # define the action space limits for unnormalization
    if centered_action:
        a_mins5d = np.array([-0.15, -0.15, -0.05, -90, 0.005])
        a_maxs5d = np.array([0.15, 0.15, 0.05, 90, 0.05])
    else:
        a_mins5d = np.array([0.56, -0.062, 0.125, -90, 0.005])
        a_maxs5d = np.array([0.7, 0.062, 0.165, 90, 0.05])

    # get past action and normalize - currently don't center normalize qpos
    qpos = np.array([0.6, 0.0, 0.165, 0.0, 0.05])
    qpos = (qpos - a_mins5d) / (a_maxs5d - a_mins5d)
    qpos = qpos * 2.0 - 1.0
    # nagent_pos = torch.from_numpy(qpos).to(torch.float32).unsqueeze(axis=0).unsqueeze(axis=0).to(device)

    # load in the goal
    raw_goal = np.load('/home/alison/Documents/GitHub/diffusion_policy_3d/goals/' + goal_str + '.npy')

    # define observation pose
    pose = fa.get_pose()
    observation_pose = np.array([0.6, 0, 0.325])
    pose.translation = observation_pose
    fa.goto_pose(pose)
    
    # initialize the n_actions counter
    n_action = 0

    # establish the list tracking how long the system takes to plan
    planning_time_list = []

    # get the observation state
    rgb2, _, pc2, _ = cam2._get_next_frame()
    rgb3, _, pc3, _ = cam3._get_next_frame()
    rgb4, _, pc4, _ = cam4._get_next_frame()
    rgb5, _, pc5, _ = cam5._get_next_frame()
    pcl, ctr = pcl_vis.unnormalize_fuse_point_clouds(pc2, pc3, pc4, pc5)
    # center and scale pointcloud
    pointcloud = (pcl - ctr) * 10

    # save the point clouds from each camera
    o3d.io.write_point_cloud(save_path + '/cam2_pcl0.ply', pc2)
    o3d.io.write_point_cloud(save_path + '/cam3_pcl0.ply', pc3)
    o3d.io.write_point_cloud(save_path + '/cam4_pcl0.ply', pc4)
    o3d.io.write_point_cloud(save_path + '/cam5_pcl0.ply', pc5)
    
    # center teh goal based on the center of the point cloud observation
    numpy_goal = (raw_goal - ctr) * 10.0
    # scale distance metric goal differently 
    # dist_goal = (raw_goal - np.mean(raw_goal, axis=0)) * 10.0
    dist_goal = numpy_goal.copy()

    # visualize observation vs goal cloud
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(pointcloud)
    pcl.colors = o3d.utility.Vector3dVector(np.tile(np.array([0,0,1]), (len(pointcloud),1)))
    goal_pcl = o3d.geometry.PointCloud()
    goal_pcl.points = o3d.utility.Vector3dVector(dist_goal)
    goal_pcl.colors = o3d.utility.Vector3dVector(np.tile(np.array([1,0,0]), (len(dist_goal),1)))
    o3d.visualization.draw_geometries([pcl, goal_pcl])

    # save observation
    np.save(save_path + '/pcl0.npy', pointcloud)
    np.save(save_path + '/center0.npy', ctr)
    cv2.imwrite(save_path + '/rgb2_state0.jpg', rgb2)
    cv2.imwrite(save_path + '/rgb3_state0.jpg', rgb3)
    cv2.imwrite(save_path + '/rgb4_state0.jpg', rgb4)
    cv2.imwrite(save_path + '/rgb5_state0.jpg', rgb5)

    # get the distance metrics between the point cloud and goal
    dist_metrics = {'CD': chamfer(pointcloud, numpy_goal),
                    'EMD': emd(pointcloud, numpy_goal),
                    'HAUSDORFF': hausdorff(pointcloud, numpy_goal)}

    print("\nDists: ", dist_metrics)
    with open(save_path + '/dist_metrics_0.txt', 'w') as f:
        f.write(str(dist_metrics))

    # for i in range(12): # maximum number of actions allowed
    max_timesteps = 8
    for t in range(max_timesteps):
        # generate the next action given the observation and goal and convert to the robot's coordinate frame
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        
        with torch.inference_mode():
            start = time.time()
            # pass the point cloud through Point-BERT to get the latent representation
            state = torch.from_numpy(pointcloud).to(torch.float32)
            states = torch.unsqueeze(state, 0).to(device)
            tokenized_states = pointbert(states)
            pcl_embed = projection_head(tokenized_states)
            pcl_embed = torch.unsqueeze(pcl_embed, 1)

            # embed goal
            goal = numpy_goal.copy()
            goal = torch.from_numpy(goal).to(torch.float32)
            goals = torch.unsqueeze(goal, 0).to(device)
            tokenized_goals = pointbert(goals)
            goal_embed = projection_head(tokenized_goals)
            goal_embed = torch.unsqueeze(goal_embed, 1) 

            # get pos_data
            if t == 0:
                pos_data = qpos
            else: 
                pos_data = prev_action
            pos_data = torch.from_numpy(pos_data).float().to(device)
            pos_data = torch.unsqueeze(pos_data, 0)

            ### query policy
            if t % query_frequency == 0:
                action_data = None
                is_pad = None
                # all_actions = policy(goal_embed, pcl_embed, action_data, is_pad, concat_goal, delta_goal, no_pos_embed)
                all_actions = policy(goal_embed, pcl_embed, pos_data, action_data, is_pad, concat_goal, delta_goal, no_pos_embed)
                print("Unsqueezed Actions", all_actions)
                all_actions = all_actions.squeeze(0).cpu().numpy()
                print("\n\nAll Actions: ", all_actions)

                # unnormalize all actions
                unnorm_as = (all_actions + 1.0) / 2.0
                unnorm_as = unnorm_as * (a_maxs5d - a_mins5d) + a_mins5d
                print("Unnormalized actions: ", unnorm_as)
                print("unnorm shape: ", unnorm_as.shape)

            # if temporal_agg:
            #     print("all time action shape: ", all_time_actions.shape)
            #     all_time_actions[[t], t:t+num_queries] = all_actions
            #     actions_for_curr_step = all_time_actions[:, t]
            #     actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            #     actions_for_curr_step = actions_for_curr_step[actions_populated]
            #     k = 0.01
            #     exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            #     exp_weights = exp_weights / exp_weights.sum()
            #     exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
            #     raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            # else:
            #     print("raw action...")
            #     raw_action = all_actions[:, t % query_frequency] 
                
        for j in range(len(all_actions)):
                

            # unnormalize action
            # pred_action = raw_action.squeeze(0).cpu().numpy()
            pred_action = unnorm_as[j,:]
            prev_action = pred_action.copy()
            end = time.time()
            planning_time_list.append(end-start)

            # execute 4 actions before replanning
            # pred_action = naction[0]
            # print("\nNormalized Predicted Action Sequence: ", pred_action)
            # action_pred = (pred_action + 1.0) / 2.0
            # action_pred = action_pred * (a_maxs5d - a_mins5d) + a_mins5d
            action_pred = pred_action
            
            print("\nUnnormalized Predicted Action Sequence: ", action_pred)
            
            # for j in range(action_pred.shape[0]):
            # unnorm_a = action_pred[j,:]
            unnorm_a = action_pred 

            if centered_action:
                print("uncentering action...")
                unnorm_a[0:3] = unnorm_a[0:3] + ctr

            print("\nAction: ", unnorm_a)

            goto_grasp(fa, unnorm_a[0], unnorm_a[1], unnorm_a[2], 0, 0, unnorm_a[3], unnorm_a[4])
            n_action+=1

            # wait here
            time.sleep(3)

            # open the gripper
            fa.open_gripper(block=True)

            # move to observation pose
            pose.translation = observation_pose
            fa.goto_pose(pose)

            # get the observation state
            rgb2, _, pc2, _ = cam2._get_next_frame()
            rgb3, _, pc3, _ = cam3._get_next_frame()
            rgb4, _, pc4, _ = cam4._get_next_frame()
            rgb5, _, pc5, _ = cam5._get_next_frame()

            # after planning should modify the center that we are using to uncenter the action!
            pcl, ctr = pcl_vis.unnormalize_fuse_point_clouds(pc2, pc3, pc4, pc5)
            # center and scale pointcloud
            pointcloud = (pcl - ctr) * 10

            # save the point clouds from each camera
            o3d.io.write_point_cloud(save_path + '/cam2_pcl' + str(t*5 + j + 1) + '.ply', pc2)
            o3d.io.write_point_cloud(save_path + '/cam3_pcl' + str(t*5 + j + 1) + '.ply', pc3)
            o3d.io.write_point_cloud(save_path + '/cam4_pcl' + str(t*5 + j + 1) + '.ply', pc4)
            o3d.io.write_point_cloud(save_path + '/cam5_pcl' + str(t*5 + j + 1) + '.ply', pc5)

            # center the goal based on this new point cloud center
            numpy_goal = (raw_goal - ctr) * 10.0
            dist_goal = numpy_goal.copy()

            # visualize observation vs goal cloud
            pcl = o3d.geometry.PointCloud()
            pcl.points = o3d.utility.Vector3dVector(pointcloud)
            pcl.colors = o3d.utility.Vector3dVector(np.tile(np.array([0,0,1]), (len(pointcloud),1)))
            goal_pcl = o3d.geometry.PointCloud()
            goal_pcl.points = o3d.utility.Vector3dVector(dist_goal)
            goal_pcl.colors = o3d.utility.Vector3dVector(np.tile(np.array([1,0,0]), (len(dist_goal),1)))
            o3d.visualization.draw_geometries([pcl, goal_pcl])

            # save observation
            np.save(save_path + '/pcl' + str(t*5 + j + 1) + '.npy', pointcloud)
            np.save(save_path + '/center' + str(t*5 + j + 1) + '.npy', ctr)
            cv2.imwrite(save_path + '/rgb2_state' + str(t*5 + j + 1) + '.jpg', rgb2)
            cv2.imwrite(save_path + '/rgb3_state' + str(t*5 + j + 1) + '.jpg', rgb3)
            cv2.imwrite(save_path + '/rgb4_state' + str(t*5 + j + 1) + '.jpg', rgb4)
            cv2.imwrite(save_path + '/rgb5_state' + str(t*5 + j + 1) + '.jpg', rgb5)

            dist_metrics = {'CD': chamfer(pointcloud, numpy_goal),
                            'EMD': emd(pointcloud, numpy_goal),
                            'HAUSDORFF': hausdorff(pointcloud, numpy_goal)}

            print("\nDists: ", dist_metrics)
            with open(save_path + '/dist_metrics_' + str(t*5 + j + 1) + '.txt', 'w') as f:
                f.write(str(dist_metrics))

            # exit loop early if the goal is reached
            if dist_metrics['CD'] < 0.07 or dist_metrics['EMD'] < 0.07:
                break
    
    # completed the experiment, send the message to the video recording loop
    done_queue.put("Done!")
    
    # create and save a dictionary of the experiment results
    results_dict = {'n_actions': n_action, 'avg planning time': np.mean(planning_time_list), 'chamfer_distance': dist_metrics['CD'], 'earth_movers_distance': dist_metrics['EMD'], 'hausdorff': dist_metrics['HAUSDORFF']}
    with open(save_path + '/results.txt', 'w') as f:
        f.write(str(results_dict))

# VIDEO THREAD
def video_loop(cam_pipeline, save_path, done_queue):
    '''
    '''
    forcc = cv2.VideoWriter_fourcc(*'XVID')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    out = cv2.VideoWriter(save_path + '/video.avi', forcc, 30.0, (1280, 800))

    frame_save_counter = 0
    # record until main loop is complete
    while done_queue.empty():
        frames = cam_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        # crop and rotate the image to just show elevated stage area
        cropped_image = color_image[320:520,430:690,:]
        rotated_image = cv2.rotate(cropped_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # save frame approx. every 100 frames
        if frame_save_counter % 100 == 0:
            # cv2.imwrite(save_path + '/external_rgb' + str(frame_save_counter) + '.jpg', rotated_image)
            cv2.imwrite(save_path + '/external_rgb' + str(frame_save_counter) + '.jpg', rotated_image)
        frame_save_counter += 1
        # out.write(rotated_image)
        out.write(color_image)
    
    cam_pipeline.stop()
    out.release()

def main(args):
    # necessary for ACT built-in requirements to run
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--temporal_agg', action='store_true')
    main(vars(parser.parse_args()))

    # -------------------------------------------------------------------
    # ---------------- Experimental Parameters to Define ----------------
    # -------------------------------------------------------------------
    exp_num = 1
    goal_shape = 'Line' # 'Cone' or 'Line' or 'X' or 'Y' or 'Cylinder
    model_path = '/home/alison/Documents/GitHub/Sculpt-ACT/checkpoints/Line_improved' # Line_fixed_aug_centered' # Line_fixed_longpred_aug_centered
    centered_action = False
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------

    exp_save = 'Experiments/Single_Line/Exp' + str(exp_num)

    # check to make sure the experiment number is not already in use, if it is, increment the number to ensure no save overwrites
    while os.path.exists(exp_save):
        exp_num += 1
        exp_save = 'Experiments/Single_Line/Exp' + str(exp_num)

    # make the experiment folder
    os.mkdir(exp_save)

    # make the experiment dictionary with important information for the experiment run
    exp_dict = {'goal: ': goal_shape,
                'model: ': model_path,
                'centered_action: ': centered_action,
                'model_path': model_path}
    
    with open(exp_save + '/experiment_params.txt', 'w') as f:
        f.write(str(exp_dict))

    # initialize the robot and reset joints
    fa = FrankaArm()
    fa.reset_joints()
    fa.open_gripper()

    # initialize the cameras
    cam2 = vis.CameraClass(2)
    cam3 = vis.CameraClass(3)
    cam4 = vis.CameraClass(4)
    cam5 = vis.CameraClass(5)

    # initialize camera 6 pipeline
    W = 1280
    H = 800
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device('152522250441')
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    pipeline.start(config)

    # initialize the 3D vision code
    pcl_vis = vis.Vision3D()    

    # load in the goal and save to the experiment folder
    goal = np.load('/home/alison/Documents/GitHub/diffusion_policy_3d/goals/' + goal_shape + '.npy')
    # center goal
    goal = (goal - np.mean(goal, axis=0)) * 10.0
    np.save(exp_save + '/goal.npy', goal)

    if goal_shape == 'X':
        t0 = np.load('/home/alison/Clay_Data/Jan27_Discrete_Demos/X/Discrete/Trajectory0/state7.npy')
        t1 = np.load('/home/alison/Clay_Data/Jan27_Discrete_Demos/X/Discrete/Trajectory1/state9.npy')
        t2 = np.load('/home/alison/Clay_Data/Jan27_Discrete_Demos/X/Discrete/Trajectory2/state7.npy')
        t3 = np.load('/home/alison/Clay_Data/Jan27_Discrete_Demos/X/Discrete/Trajectory3/state7.npy')
        t4 = np.load('/home/alison/Clay_Data/Jan27_Discrete_Demos/X/Discrete/Trajectory4/state6.npy')
    elif goal_shape == 'Line':
        t0 = np.load('/home/alison/Clay_Data/Jan27_Discrete_Demos/Line/Discrete/Trajectory0/state6.npy')
        t1 = np.load('/home/alison/Clay_Data/Jan27_Discrete_Demos/Line/Discrete/Trajectory1/state5.npy')
        t2 = np.load('/home/alison/Clay_Data/Jan27_Discrete_Demos/Line/Discrete/Trajectory2/state5.npy')
        t3 = np.load('/home/alison/Clay_Data/Jan27_Discrete_Demos/Line/Discrete/Trajectory3/state6.npy')
        t4 = np.load('/home/alison/Clay_Data/Jan27_Discrete_Demos/Line/Discrete/Trajectory4/state5.npy')
    elif goal_shape == 'Cone':
        t0 = np.load('/home/alison/Clay_Data/Jan27_Discrete_Demos/Cone/Discrete/Trajectory0/state10.npy')
        t1 = np.load('/home/alison/Clay_Data/Jan27_Discrete_Demos/Cone/Discrete/Trajectory1/state11.npy')
        t2 = np.load('/home/alison/Clay_Data/Jan27_Discrete_Demos/Cone/Discrete/Trajectory2/state8.npy')
        t3 = np.load('/home/alison/Clay_Data/Jan27_Discrete_Demos/Cone/Discrete/Trajectory3/state7.npy')
        t4 = np.load('/home/alison/Clay_Data/Jan27_Discrete_Demos/Cone/Discrete/Trajectory4/state9.npy')
    
    t0 = (t0 - np.mean(t0, axis=0)) * 10.0
    t1 = (t1 - np.mean(t1, axis=0)) * 10.0
    t2 = (t2 - np.mean(t2, axis=0)) * 10.0
    t3 = (t3 - np.mean(t3, axis=0)) * 10.0
    t4 = (t4 - np.mean(t4, axis=0)) * 10.0

    # initialize the threads
    done_queue = queue.Queue()

    main_thread = threading.Thread(target=experiment_loop, args=(fa, cam2, cam3, cam4, cam5, pcl_vis, exp_save, goal_shape, model_path, done_queue, centered_action))
    video_thread = threading.Thread(target=video_loop, args=(pipeline, exp_save, done_queue))

    main_thread.start()
    video_thread.start()