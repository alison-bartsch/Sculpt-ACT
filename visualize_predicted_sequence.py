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
from embeddings.embeddings import EncoderHead, EncoderHeadFiLM
from pointBERT.tools import builder
from pointBERT.utils.config import cfg_from_yaml_file

import json
from os.path import join

import open3d as o3d

from imitate_clay_episodes import *
from action_geometry_utils import *

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

def visualize_grasp(state, next_state, action):
    action = unnormalize_a(action)

    # center action at origin of the point cloud
    pcl_center = np.array([0.6, 0.0, 0.25])
    action[0:3] = action[0:3] - pcl_center
    action[0:3] = action[0:3]

    # scale the action (multiply x,y,z,d by 10)
    action_scaled = action * 10
    action_scaled[3] = action[3] # don't scale the rotation
    len = 10 * 0.1

    # get the points and lines for the action orientation visualization
    ctr = action_scaled[0:3]
    upper_ctr = ctr + np.array([0,0, 0.6])
    rz = 90 + action_scaled[3]
    points, lines = line_3d_start_end(ctr, rz, len)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (lines.shape[0],1)))

    # get end points
    delta = 1.0 - action_scaled[4]
    end_pts, _ = line_3d_start_end(ctr, rz, len - delta)
    top_end_pts, _ = line_3d_start_end(upper_ctr, rz, len - delta)

    # get the top points for the grasp (given gripper finger height)
    top_points, _ = line_3d_start_end(upper_ctr, rz, len)

    # gripper 1 
    g1_base_start, _ = line_3d_start_end(points[0], rz+90, 0.18)
    g1_base_end, _ = line_3d_start_end(end_pts[0], rz+90, 0.18)
    g1_top_start, _ = line_3d_start_end(top_points[0], rz+90, 0.18)
    g1_top_end, _ = line_3d_start_end(top_end_pts[0], rz+90, 0.18)
    g1_points, g1_lines = line_3d_point_set([g1_base_start, g1_base_end, g1_top_start, g1_top_end])

    # create oriented bounding box
    g1_test = o3d.geometry.OrientedBoundingBox()
    g1_bbox = g1_test.create_from_points(o3d.utility.Vector3dVector(g1_points))
    g1_idx = g1_bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(state))

    # create g1 
    g1 = o3d.geometry.LineSet()
    g1.points = o3d.utility.Vector3dVector(g1_points)
    g1.lines = o3d.utility.Vector2iVector(g1_lines)
    g1.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (g1_lines.shape[0],1)))

    # gripper 2
    g2_base_start, _ = line_3d_start_end(points[1], rz+90, 0.18)
    g2_base_end, _ = line_3d_start_end(end_pts[1], rz+90, 0.18)
    g2_top_start, _ = line_3d_start_end(top_points[1], rz+90, 0.18)
    g2_top_end, _ = line_3d_start_end(top_end_pts[1], rz+90, 0.18)
    g2_points, g2_lines = line_3d_point_set([g2_base_start, g2_base_end, g2_top_start, g2_top_end])

    # create oriented bounding box
    g2_test = o3d.geometry.OrientedBoundingBox()
    g2_bbox = g2_test.create_from_points(o3d.utility.Vector3dVector(g2_points))
    g2_idx = g2_bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(state))

    # create g2
    g2 = o3d.geometry.LineSet()
    g2.points = o3d.utility.Vector3dVector(g2_points)
    g2.lines = o3d.utility.Vector2iVector(g2_lines)
    g2.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (g2_lines.shape[0],1)))

    # create state next state pointclouds
    og_pcl = o3d.geometry.PointCloud()
    og_pcl.points = o3d.utility.Vector3dVector(state)
    og_colors = np.tile(np.array([0, 0, 1]), (state.shape[0],1))
    og_pcl.colors = o3d.utility.Vector3dVector(og_colors)

    g_pcl = o3d.geometry.PointCloud()
    g_pcl.points = o3d.utility.Vector3dVector(next_state)
    g_colors = np.tile(np.array([0, 1, 0]), (next_state.shape[0],1))
    g_pcl.colors = o3d.utility.Vector3dVector(g_colors)

    # o3d.visualization.draw_geometries([og_pcl, g_pcl, ctr_action, line_set, g1, g2, g1_bbox])
    o3d.visualization.draw_geometries([og_pcl, g_pcl, line_set, g1, g2, g1_bbox])


def visualize_pred_action_sequence(pred_actions, state, goal):
    pred_actions = pred_actions.cpu().detach().numpy()
    state = state.detach().numpy()
    goal = goal.detach().numpy()

    # iterate through the predicted actions
    for i in range(pred_actions.shape[0]):
        # unnormalize the action
        action = pred_actions[i][0]
        print("action: ", action)
        action = unnormalize_a(action)

        # center action at origin of the point cloud
        pcl_center = np.array([0.6, 0.0, 0.25])
        action[0:3] = action[0:3] - pcl_center
        action[0:3] = action[0:3]

        # scale the action (multiply x,y,z,d by 10)
        action_scaled = action * 10
        action_scaled[3] = action[3] # don't scale the rotation
        len = 10 * 0.1

        # get the points and lines for the action orientation visualization
        ctr = action_scaled[0:3]
        upper_ctr = ctr + np.array([0,0, 0.6])
        rz = 90 + action_scaled[3]
        points, lines = line_3d_start_end(ctr, rz, len)
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (lines.shape[0],1)))

        # get end points
        delta = 1.0 - action_scaled[4]
        end_pts, _ = line_3d_start_end(ctr, rz, len - delta)
        top_end_pts, _ = line_3d_start_end(upper_ctr, rz, len - delta)

        # get the top points for the grasp (given gripper finger height)
        top_points, _ = line_3d_start_end(upper_ctr, rz, len)

        # gripper 1 
        g1_base_start, _ = line_3d_start_end(points[0], rz+90, 0.18)
        g1_base_end, _ = line_3d_start_end(end_pts[0], rz+90, 0.18)
        g1_top_start, _ = line_3d_start_end(top_points[0], rz+90, 0.18)
        g1_top_end, _ = line_3d_start_end(top_end_pts[0], rz+90, 0.18)
        g1_points, g1_lines = line_3d_point_set([g1_base_start, g1_base_end, g1_top_start, g1_top_end])

        # create oriented bounding box
        g1_test = o3d.geometry.OrientedBoundingBox()
        g1_bbox = g1_test.create_from_points(o3d.utility.Vector3dVector(g1_points))
        g1_idx = g1_bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(state))

        inlier_pts = state.copy()

        # pointcloud with points inside rectangle
        g1_inside = state[g1_idx,:]
        g1_inside_pcl = o3d.geometry.PointCloud()
        g1_inside_pcl.points = o3d.utility.Vector3dVector(g1_inside)
        g1_inside_colors = np.tile(np.array([1, 0, 0]), (g1_inside.shape[0],1))
        g1_inside_pcl.colors = o3d.utility.Vector3dVector(g1_inside_colors)

        # get the displacement vector for the gripper 1 base
        g1_dir_unit = dir_vec_from_points(end_pts[0], points[0])
        g1_displacement_vec = end_pts[0] - points[0]

        # apply the displacement vector to all the points in the state point cloud
        g1_diffs = np.tile(end_pts[0], (inlier_pts[g1_idx,:].shape[0],1)) - inlier_pts[g1_idx,:] 
        g1_diffs = np.linalg.norm(g1_diffs, axis=1)
        inlier_pts[g1_idx,:] = inlier_pts[g1_idx,:] -  np.tile(g1_diffs, (3,1)).T * np.tile(g1_dir_unit, (inlier_pts[g1_idx,:].shape[0],1))

        g1 = o3d.geometry.LineSet()
        g1.points = o3d.utility.Vector3dVector(g1_points)
        g1.lines = o3d.utility.Vector2iVector(g1_lines)
        g1.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (g1_lines.shape[0],1)))

        # gripper 2
        g2_base_start, _ = line_3d_start_end(points[1], rz+90, 0.18)
        g2_base_end, _ = line_3d_start_end(end_pts[1], rz+90, 0.18)
        g2_top_start, _ = line_3d_start_end(top_points[1], rz+90, 0.18)
        g2_top_end, _ = line_3d_start_end(top_end_pts[1], rz+90, 0.18)
        g2_points, g2_lines = line_3d_point_set([g2_base_start, g2_base_end, g2_top_start, g2_top_end])

        # create oriented bounding box
        g2_test = o3d.geometry.OrientedBoundingBox()
        g2_bbox = g2_test.create_from_points(o3d.utility.Vector3dVector(g2_points))
        g2_idx = g2_bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(state))
        
        # pointcloud with points inside rectangle
        g2_inside = state[g2_idx,:]
        g2_inside_pcl = o3d.geometry.PointCloud()
        g2_inside_pcl.points = o3d.utility.Vector3dVector(g2_inside)
        g2_inside_colors = np.tile(np.array([1, 0, 0]), (g2_inside.shape[0],1))
        g2_inside_pcl.colors = o3d.utility.Vector3dVector(g2_inside_colors)

        # get the displacement vector for the gripper 1 base
        g2_dir_unit = dir_vec_from_points(end_pts[1], points[1])
        g2_displacement_vec = end_pts[1] - points[1]

        # apply the displacement vector to all the points in the state point cloud
        g2_diffs = np.tile(end_pts[1], (inlier_pts[g2_idx,:].shape[0],1)) - inlier_pts[g2_idx,:] 
        g2_diffs = np.linalg.norm(g2_diffs, axis=1)
        
        inlier_pts[g2_idx,:] = inlier_pts[g2_idx,:] -  np.tile(g2_diffs, (3,1)).T * np.tile(g2_dir_unit, (inlier_pts[g2_idx,:].shape[0],1))
        inliers = o3d.geometry.PointCloud()
        inliers.points = o3d.utility.Vector3dVector(inlier_pts)
        inlier_colors = np.tile(np.array([1, 0, 0]), (inlier_pts.shape[0],1))
        inliers.colors = o3d.utility.Vector3dVector(inlier_colors)

        g2 = o3d.geometry.LineSet()
        g2.points = o3d.utility.Vector3dVector(g2_points)
        g2.lines = o3d.utility.Vector2iVector(g2_lines)
        g2.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (g2_lines.shape[0],1)))

        # test plot the point cloud and action and goal
        og_pcl = o3d.geometry.PointCloud()
        og_pcl.points = o3d.utility.Vector3dVector(state)
        og_colors = np.tile(np.array([0, 0, 1]), (state.shape[0],1))
        og_pcl.colors = o3d.utility.Vector3dVector(og_colors)

        g_pcl = o3d.geometry.PointCloud()
        g_pcl.points = o3d.utility.Vector3dVector(goal)
        g_colors = np.tile(np.array([0, 1, 0]), (goal.shape[0],1))
        g_pcl.colors = o3d.utility.Vector3dVector(g_colors)

        # create black point cloud of g1_points and g2_points to sanity check corners
        corners = o3d.geometry.PointCloud()
        corners.points = o3d.utility.Vector3dVector(np.concatenate((g1_points, g2_points), axis=0))
        corners_colors = np.tile(np.array([0, 0, 0]), (np.concatenate((g1_points, g2_points)).shape[0],1))
        corners.colors = o3d.utility.Vector3dVector(corners_colors)

        ctr_action = o3d.geometry.PointCloud()
        action_cloud = action_scaled[0:3].reshape(1,3)
        print("Action: ", action_cloud)
        ctr_action.points = o3d.utility.Vector3dVector(action_scaled[0:3].reshape(1,3))
        ctr_colors = np.tile(np.array([1, 0, 0]), (1,1))
        ctr_action.colors = o3d.utility.Vector3dVector(ctr_colors)
        o3d.visualization.draw_geometries([og_pcl, g_pcl, ctr_action, line_set, g1, g2, corners, g1_bbox])
        o3d.visualization.draw_geometries([og_pcl, g_pcl, ctr_action, line_set, g1, g2, g1_inside_pcl, g2_inside_pcl, corners])
        o3d.visualization.draw_geometries([g_pcl, ctr_action, line_set, g1, g2, inliers, corners])

        # TODO: iterate with the predicted cloud for the next action in sequence to get coarse prediction of final shape?

        # convert the normalized action to real-world frame
        # visualize the grasp given the world frame (will need to center about (0,0,0))
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
    backbone = 'resnet18' 
    if policy_class == 'ACT':
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
        deploy_model(exp_config, ckpt_name, save_episode=True)


def deploy_model(config, ckpt_name, save_episode=True):
    # import a starting state from the dataset
    pointcloud = np.load('/home/alison/Clay_Data/Raw_Data/Clay_Demo_Trajectories/X/Trajectory4/State1.npy')
    pointcloud = recenter_pcl(pointcloud)

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

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')

    # load point-BERT
    device = torch.device('cuda')
    if pre_trained_encoder:
        enc_checkpoint = torch.load('pointBERT/encoder_weights/checkpoint', map_location=torch.device('cpu'))
        encoder_head = enc_checkpoint['encoder_head'].to(device)
    else:
        encoded_dim = 768
        latent_dim = 512
        if film_goal:
            encoder_head = EncoderHeadFiLM(encoded_dim, latent_dim, encoded_dim).to(device)
        else:
            encoder_head = EncoderHead(encoded_dim, latent_dim).to(device)

    config = cfg_from_yaml_file('pointBERT/cfgs/PointTransformer.yaml')
    model_config = config.model
    pointbert = builder.model_builder(model_config)
    weights_path = 'pointBERT/point-BERT-weights/Point-BERT.pth'
    pointbert.load_model_from_ckpt(weights_path)
    pointbert.to(device)

    # import the goal point cloud
    goal = np.load('X_target.npy')
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
        goal_embed = encoder_head(tokenized_goals)
        goal_embed = torch.unsqueeze(goal_embed, 1) 

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks
    num_rollouts = 1
    for rollout_id in range(num_rollouts):
        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        with torch.inference_mode():
            for t in range(max_timesteps):
                # pass the point cloud through Point-BERT to get the latent representation
                state = torch.from_numpy(pointcloud).to(torch.float32)
                states = torch.unsqueeze(state, 0).to(device)
                tokenized_states = pointbert(states)
                if film_goal:
                    pcl_embed = encoder_head(tokenized_states, tokenized_goals)
                else:
                    pcl_embed = encoder_head(tokenized_states)
                pcl_embed = torch.unsqueeze(pcl_embed, 1) 

                ### query policy
                if t % query_frequency == 0:
                    action_data = None
                    is_pad = None
                    all_actions = policy(goal_embed, pcl_embed, action_data, is_pad, concat_goal, delta_goal, no_pos_embed)
                    print("all actions shape: ", all_actions.shape)
                    print("all actions: ", all_actions)
                    # assert False
                    visualize_pred_action_sequence(all_actions, state, goal)
                if temporal_agg:
                    print("all time action shape: ", all_time_actions.shape)

                    all_time_actions[[t], t:t+num_queries] = all_actions
                    print("all time actions: ", all_time_actions)
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    visualize_pred_action_sequence(raw_action, state, goal)
                else:
                    print("raw action...")
                    raw_action = all_actions[:, t % query_frequency] 
                    print("Raw action shape: ", torch.unsqueeze(raw_action, dim=1).shape)
                    visualize_pred_action_sequence(torch.unsqueeze(raw_action, dim=1), state, goal)

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
    parser.add_argument('--temporal_agg', action='store_true')

    main(vars(parser.parse_args()))
