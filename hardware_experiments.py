import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data, load_clay_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import robomail.vision as vis
from robot_utils import *
# from dynamics.dynamics_model import EncoderHead
from embeddings.embeddings import EncoderHead, EncoderHeadFiLM, EncoderHeadFiLMPretrained
from pointBERT.tools import builder
from pointBERT.utils.config import cfg_from_yaml_file

import json
from os.path import join

import open3d as o3d

from frankapy import FrankaArm
from imitate_clay_episodes import *

from emd import earth_mover_distance
from pytorch3d.loss import chamfer_distance

import IPython
e = IPython.embed

def recenter_pcl(pointcloud):
    """
    Assume pcl is a numpy array.
    """
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(pointcloud)
    goal_ctr = pcl.get_center()
    return pointcloud - goal_ctr

def visualize_pred_action_sequence(pred_actions_list, goal):
    pass

def main(args):
    set_seed(1)

    # ckpt_dir, policy_class, task_name, tempoeral_agg

    # command line parameters
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    episode_len = 6 # maximum episode lengths

    # fixed parameters
    state_dim = 5 #14
    lr_backbone = 1e-5
    backbone = 'resnet18' # TODO: FIX THIS!
    if policy_class == 'ACT':
        # enc_layers = 4
        # dec_layers = 7
        # nheads = 8
        # policy_config = {'lr': args['lr'],
        #                  'num_queries': args['chunk_size'],
        #                  'kl_weight': args['kl_weight'],
        #                  'hidden_dim': args['hidden_dim'],
        #                  'dim_feedforward': args['dim_feedforward'],
        #                  'lr_backbone': lr_backbone,
        #                  'backbone': backbone,
        #                  'enc_layers': enc_layers,
        #                  'dec_layers': dec_layers,
        #                  'nheads': nheads,
        #                 #  'camera_names': camera_names,
        #                  }
        with open(ckpt_dir + '/policy_config.json') as json_file:
            policy_config = json.load(json_file)

    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,}
    else:
        raise NotImplementedError
    
    print("\n")

    exp_config = {
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'policy_class': policy_class,
        'policy_config': policy_config,
        'temporal_agg': args['temporal_agg'],
    }

    ckpt_names = [f'policy_best.ckpt']
    for ckpt_name in ckpt_names:
        harware_eval(exp_config, ckpt_name, save_episode=True)


# def make_policy(policy_class, policy_config):
#     if policy_class == 'ACT':
#         # print("\nPolicy Config: ", policy_config)
#         policy = ACTPolicy(policy_config)
#     elif policy_class == 'CNNMLP':
#         policy = CNNMLPPolicy(policy_config)
#     else:
#         raise NotImplementedError
#     return policy


# def make_optimizer(policy_class, policy):
#     if policy_class == 'ACT':
#         optimizer = policy.configure_optimizers()
#     elif policy_class == 'CNNMLP':
#         optimizer = policy.configure_optimizers()
#     else:
#         raise NotImplementedError
#     return optimizer


# def get_image(ts, camera_names):
#     curr_images = []
#     for cam_name in camera_names:
#         curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
#         curr_images.append(curr_image)
#     curr_image = np.stack(curr_images, axis=0)
#     curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
#     return curr_image

def harware_eval(config, ckpt_name, save_episode=True):
    # initialize the robot and reset joints
    fa = FrankaArm()

    # initialize the cameras
    cam2 = vis.CameraClass(2)
    cam3 = vis.CameraClass(3)
    cam4 = vis.CameraClass(4)
    cam5 = vis.CameraClass(5)

    # initialize the 3D vision code
    pcl_vis = vis.Vision3D()

    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    max_timesteps = config['episode_len']
    temporal_agg = config['temporal_agg']

    # load in the modifiers from params.json based on ckpt_dir
    with open(ckpt_dir + '/params.json') as json_file:
        params_config = json.load(json_file)

    concat_goal = params_config['concat_goal']
    delta_goal = params_config['delta_goal']
    film_goal = params_config['film_goal']
    no_pos_embed = params_config['no_pos_embed']
    stopping_action = params_config['stopping_action']
    pre_trained_encoder = params_config['pre_trained_encoder']

    # experiment config
    # experiment_config = params_config.update(config) # THIS ISN'T WORKING

    # add config to params_config
    experiment_config = params_config
    experiment_config['ckpt_dir'] = ckpt_dir
    experiment_config['state_dim'] = state_dim
    experiment_config['policy_class'] = policy_class
    experiment_config['policy_config'] = policy_config
    experiment_config['episode_len'] = max_timesteps
    experiment_config['temporal_agg'] = temporal_agg

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')

    # # load point-BERT
    # device = torch.device('cuda')
    # enc_checkpoint = torch.load('pointBERT/encoder_weights/checkpoint', map_location=torch.device('cpu'))
    # encoder_head = enc_checkpoint['encoder_head'].to(device)
    # pointbert_config = cfg_from_yaml_file('pointBERT/cfgs/PointTransformer.yaml')
    # model_config = pointbert_config.model
    # pointbert = builder.model_builder(model_config)
    # weights_path = 'pointBERT/point-BERT-weights/Point-BERT.pth'
    # pointbert.load_model_from_ckpt(weights_path)
    # pointbert.to(device)

    # # load point-BERT from file in ckpt_dir
    # device = torch.device('cuda')
    # enc_checkpoint = torch.load(ckpt_dir + '/encoder_best_checkpoint.zip', map_location=torch.device('cpu'))
    # encoder_head = enc_checkpoint['encoder_head'].to(device)
    # pointbert_config = cfg_from_yaml_file('pointBERT/cfgs/PointTransformer.yaml')
    # model_config = pointbert_config.model
    # pointbert = builder.model_builder(model_config)
    # weights_path = 'pointBERT/point-BERT-weights/Point-BERT.pth'
    # pointbert.load_model_from_ckpt(weights_path)
    # pointbert.to(device)
    # # config = cfg_from_yaml_file('pointBERT/cfgs/PointTransformer.yaml')
    # # model_config = config.model
    # # pointbert = builder.model_builder(model_config)
    # # weights_path = ckpt_dir + '/best_pointbert.pth' # 'pointBERT/point-BERT-weights/Point-BERT.pth'
    # # pointbert.load_model_from_ckpt(weights_path)
    # # pointbert.to(device)

    # load point-BERT
    device = torch.device('cuda')
    if pre_trained_encoder:
        enc_checkpoint = torch.load('pointBERT/encoder_weights/checkpoint', map_location=torch.device('cpu'))
        projection_head = enc_checkpoint['encoder_head'].to(device)
        print("loaded encoder head...")
    else:
        encoded_dim = 768
        latent_dim = 512
        if film_goal:
            # encoder_head = EncoderHeadFiLM(encoded_dim, latent_dim, encoded_dim).to(device)
            projection_head = EncoderHeadFiLMPretrained(encoded_dim, latent_dim, projection_head, encoded_dim).to(device)
        else:
            projection_head = EncoderHead(encoded_dim, latent_dim).to(device)

    config = cfg_from_yaml_file('pointBERT/cfgs/PointTransformer.yaml')
    model_config = config.model
    pointbert = builder.model_builder(model_config)
    weights_path = 'pointBERT/point-BERT-weights/Point-BERT.pth'
    pointbert.load_model_from_ckpt(weights_path)
    pointbert.to(device)

    # import the goal point cloud
    goal = np.load('X_target.npy')

    # reprocess goal if collected before new calibration
    goal = recenter_pcl(goal)

    # create open3d goal object for visualization
    goal_o3d = o3d.geometry.PointCloud()
    goal_o3d.points = o3d.utility.Vector3dVector(goal)
    goal_o3d_colors = np.tile(np.array([1,0,0]), (len(goal),1))
    goal_o3d.colors = o3d.utility.Vector3dVector(goal_o3d_colors)

    # get goal embedding once
    goal = torch.from_numpy(goal).to(torch.float32)
    goals = torch.unsqueeze(goal, 0).to(device)
    tokenized_goals = pointbert(goals)
    if film_goal:
        goal_embed = None
    else:
        goal_embed = projection_head(tokenized_goals)
        goal_embed = torch.unsqueeze(goal_embed, 1) 

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks
    num_rollouts = 5
    for rollout_id in range(num_rollouts):
        # create experiment folder
        exp_name = 'exp26'
        os.mkdir('Experiments/' + exp_name)

        # save the config file
        with open(join('Experiments/' + exp_name + '/', 'experiment_params.json'), 'w') as f:
            json.dump(experiment_config, f)

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        with torch.inference_mode():
            fa.reset_joints()
            fa.reset_pose()
            fa.open_gripper()

            # move to observation pose
            observation_pose = np.array([0.6, 0, 0.325])
            pose = fa.get_pose()
            pose.translation = observation_pose
            fa.goto_pose(pose)

            # TODO: loop checking for improvement
            # get the observation and calculate cd and emd
            # prev_cd = 0.01
            # prev_emd = 0.1
            # while improving:
                # do the stuff
                # get the next state and calculate cd and emd
                # cd = chamfer_distance(target, cur_state)[0].cpu().detach().numpy()
                # emd = earth_mover_distance(target, cur_state, transpose=False)
                # if (prev_cd - curr_cd) > 0.001 or (prev_emd - curr_emd) > 0.01:
                    # improving = False
            
            # TODO: loop checking for stopping token
            # while not stopping_token:
                # do the stuff
                # stopping_token = bool(1 / (1 + exp(-action[-1]))

            for t in range(max_timesteps):
            # while improving:
                # get the current state (point cloud)
                _, _, pc2, _ = cam2._get_next_frame()
                _, _, pc3, _ = cam3._get_next_frame()
                _, _, pc4, _ = cam4._get_next_frame()
                _, _, pc5, _ = cam5._get_next_frame()
                pointcloud = pcl_vis.fuse_point_clouds(pc2, pc3, pc4, pc5, vis=False)

                # save the raw point clouds
                os.mkdir('Experiments/' + exp_name + '/Raw_Pointclouds' + str(t)) 
                o3d.io.write_point_cloud('Experiments/' + exp_name + '/Raw_Pointclouds' + str(t) + '/cam2.ply', pc2) 
                o3d.io.write_point_cloud('Experiments/' + exp_name + '/Raw_Pointclouds' + str(t) + '/cam3.ply', pc3)
                o3d.io.write_point_cloud('Experiments/' + exp_name + '/Raw_Pointclouds' + str(t) + '/cam4.ply', pc4)
                o3d.io.write_point_cloud('Experiments/' + exp_name + '/Raw_Pointclouds' + str(t) + '/cam5.ply', pc5)

                # TODO: pointcloud, center = use pcl_vis.unnormalize_fuse_point_clouds() to have center to change the action scaling 

                # visualize the state and target and save as png 
                pcl_o3d = o3d.geometry.PointCloud()
                pcl_o3d.points = o3d.utility.Vector3dVector(pointcloud)
                pcl_o3d_colors = np.tile(np.array([0,0,1]), (len(pointcloud),1))
                pcl_o3d.colors = o3d.utility.Vector3dVector(pcl_o3d_colors)
                pcl_vis.pcl_to_image(pcl_o3d, goal_o3d, 'Experiments/' + exp_name + '/pointclouds' + str(t) + '.png')


                # pass the point cloud through Point-BERT to get the latent representation
                state = torch.from_numpy(pointcloud).to(torch.float32)
                states = torch.unsqueeze(state, 0).to(device)
                tokenized_states = pointbert(states)
                if film_goal:
                    pcl_embed = projection_head(tokenized_states)
                    # pcl_embed = projection_head.forward(tokenized_states, tokenized_goals)
                else:
                    pcl_embed = projection_head(tokenized_states)
                pcl_embed = torch.unsqueeze(pcl_embed, 1) 

                ### query policy
                if t % query_frequency == 0:
                    # all_actions = policy(qpos, pcl_embed)
                    # all_actions = policy(goal_embed, pcl_embed)
                    action_data = None
                    is_pad = None
                    all_actions = policy(goal_embed, pcl_embed, action_data, is_pad, concat_goal, delta_goal, no_pos_embed)
                if temporal_agg:
                    print("all time action shape: ", all_time_actions.shape)

                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    print("raw action...")
                    raw_action = all_actions[:, t % query_frequency] 

                ### post-process actions
                print("Raw action: ", raw_action)
                raw_action = raw_action.squeeze(0).cpu().numpy()
                # convert raw action into robot action space
                robot_action = get_real_action_from_normalized(raw_action)
                # execute the grasp action and print the action
                goto_grasp(fa, robot_action[0], robot_action[1], robot_action[2], 0, 0, robot_action[3], robot_action[4])
                print("\nGRASP ACTION: ", robot_action)

                # wait here
                time.sleep(3)

                # open the gripper
                fa.open_gripper(block=True)
                # time.sleep(2)

                # move to observation pose
                pose.translation = observation_pose
                fa.goto_pose(pose)

        # wait for ENTER key for the clay to be reset
        input("Press ENTER to continue when clay has been reset...")


# def forward_pass(data, policy, pointbert, encoder_head):
#     """
#     Version of forward pass with goal conditioning.
#     """
#     goal_data, state_data, action_data, is_pad = data
#     goal_data, state_data, action_data, is_pad = goal_data.cuda(), state_data.cuda(), action_data.cuda(), is_pad.cuda()

#     state_data = state_data.to(torch.float32)
#     tokenized_states = pointbert(state_data)
#     pcl_embed = encoder_head(tokenized_states)
#     pcl_embed = torch.unsqueeze(pcl_embed, 1)

#     goal_data = goal_data.to(torch.float32)
#     tokenized_goals = pointbert(goal_data)
#     goal_embed = encoder_head(tokenized_goals)
#     goal_embed = torch.unsqueeze(goal_embed, 1)

#     return policy(goal_embed, pcl_embed, action_data, is_pad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--eval', action='store_true')
    # parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    # parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    # parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    # parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    # parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    # parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    # parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    # args = parser.parse_args()
    # print("args: ", args)

    # args_dict = vars(args)
    # print("vars: ", args_dict)
    # main(args_dict)
    # # assert False
    
    main(vars(parser.parse_args()))
    # main(vars(args))
