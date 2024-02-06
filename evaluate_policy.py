import torch
import numpy as np
import os
import argparse

from robot_utils import *
# from dynamics.dynamics_model import EncoderHead
from embeddings.embeddings import EncoderHead, EncoderHeadFiLM
from pointBERT.tools import builder
from pointBERT.utils.config import cfg_from_yaml_file

import json
from os.path import join

import open3d as o3d

from imitate_clay_episodes import *
# from action_geometry_utils import *

import IPython
e = IPython.embed
from vis_utils import *


# load in trained model
# load in experiment parameters from .ipynb file
# load in the dataset path
# list of N trajectories to visualize

# for each trajectory:
    # load in the state, goal, action sequence
    # pass these through the model to get the predicted action sequence

    # for i in range(pre_action_seq.shape[1]):
        # visualize the original action
        # visualize the predicted action

def main(args):
    set_seed(1)

    # ckpt_dir, policy_class, task_name, tempoeral_agg

    # command line parameters
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    episode_len = 6 # maximum episode lengths

    # fixed parameters
    state_dim = 5 #14  #TODO modify if centered action 
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

    # Trajectories
    # trajs = [1, 4, 8, 9]
    trajs = [3,6,9]
    # starting_states = [0, 1, 3]
    starting_states = [0]
    dataset_path = '/home/alison/Clay_Data/Trajectory_Data/No_Aug_Dec14_Human_Demos/X'

    ckpt_names = [f'policy_best.ckpt']
    for ckpt_name in ckpt_names:
        # load in model
        set_seed(1000)
        # ckpt_dir = config['ckpt_dir']
        # state_dim = config['state_dim']
        # policy_class = config['policy_class']
        # policy_config = config['policy_config']
        # max_timesteps = config['episode_len']
        # temporal_agg = config['temporal_agg']

        # load in the modifiers from params.json based on ckpt_dir
        with open(ckpt_dir + '/params.json') as json_file:
            params_config = json.load(json_file)

        concat_goal = params_config['concat_goal']
        delta_goal = params_config['delta_goal']
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
        enc_checkpoint = torch.load(ckpt_dir + '/encoder_best_checkpoint', map_location=torch.device('cpu'))
        encoder_head = enc_checkpoint['encoder_head'].to(device)

        config = cfg_from_yaml_file('pointBERT/cfgs/PointTransformer.yaml')
        model_config = config.model
        pointbert = builder.model_builder(model_config)
        # weights_path = 'pointBERT/point-BERT-weights/Point-BERT.pth'
        weights_path = ckpt_dir + '/best_pointbert.pth'
        pointbert.load_model_from_ckpt(weights_path)
        pointbert.to(device)

        # iterate through the trajectories
        for traj in trajs:
            path = dataset_path + '/Trajectory' + str(traj)

            # setup goal
            goal_unctr = np.load(path + '/goal.npy')
            print("goal ctr mean: ", np.mean(goal_unctr, axis=0))

            # center goal
            goal = (goal_unctr - np.mean(goal_unctr, axis=0)) * 10.0

            # # create open3d goal object for visualization
            # goal_o3d = o3d.geometry.PointCloud()
            # goal_o3d.points = o3d.utility.Vector3dVector(goal)
            # goal_o3d_colors = np.tile(np.array([1,0,0]), (len(goal),1))
            # goal_o3d.colors = o3d.utility.Vector3dVector(goal_o3d_colors)

            # get goal embedding once
            goal = torch.from_numpy(goal).to(torch.float32)
            goals = torch.unsqueeze(goal, 0).to(device)
            tokenized_goals = pointbert(goals)
            goal_embed = encoder_head(tokenized_goals)
            goal_embed = torch.unsqueeze(goal_embed, 1) 

            # iterate through the starting states
            for s in starting_states:
                state = np.load(path + '/state' + str(s) + '.npy')

                # uncenter and unscale state for visualization
                ctr = np.load(path + '/pcl_center' + str(s) + '.npy')
                state_unctr = state * 0.1 + ctr

                # recreate actual test-time inference
                with torch.inference_mode():
                    # pass the point cloud through Point-BERT to get the latent representation
                    state = torch.from_numpy(state).to(torch.float32)
                    states = torch.unsqueeze(state, 0).to(device)
                    tokenized_states = pointbert(states)
                    pcl_embed = encoder_head(tokenized_states)
                    pcl_embed = torch.unsqueeze(pcl_embed, 1) 

                    ### query policy
                    action_data = None
                    is_pad = None
                    all_actions = policy(goal_embed, pcl_embed, action_data, is_pad, concat_goal, delta_goal, no_pos_embed)
                    # print("all actions shape: ", all_actions.shape)
                    # print("all actions: ", all_actions)
                    print(all_actions[0][0])
                    
                gt_actions = []
                n_actions = 5 # TODO: get this from the dataset
                for i in range(1,n_actions+1):
                    action = np.load(path + '/unnormalized_action' + str(i) + '.npy')
                    gt_actions.append(action)
                # gt_actions = np.expand_dims(np.array(gt_actions), axis=0)
                # print("GT ACTIONS <SHAPE: ", gt_actions.shape)
                    
                # # recreate training-time inference 
                # with torch.inference_mode():
                #     ### query policy
                #     action_data = torch.from_numpy(np.expand_dims(np.array(gt_actions), axis=0)).to(torch.float32).to(device)
                #     is_pad = None
                #     recon_actions = policy(goal_embed, pcl_embed, action_data, is_pad, concat_goal, delta_goal, no_pos_embed)
                #     print("all actions shape: ", all_actions.shape)
                #     print("all actions: ", all_actions)
                #     print(recon_actions[0][0])
            
                for i in range(n_actions):
                    print("\n\nPredictions visualizations")
                    print("GT action: ", gt_actions[i])
                    # visualize the original action
                    vis_grasp(state_unctr, goal_unctr, gt_actions[i], offset=False)

                    # # visualize the reconstructed action
                    # recon_a = recon_actions[0][i].cpu().detach().numpy()
                    # a_mins5d = np.array([0.55, -0.035, 0.19, -90, 0.005])
                    # a_maxs5d = np.array([0.63, 0.035, 0.25, 90, 0.05])
                    # unnorm_recon_a = recon_a * (a_maxs5d - a_mins5d) + a_mins5d
                    # vis_grasp(state_unctr, goal_unctr, unnorm_recon_a, offset=False)
                    
                    # visualize the predicted action
                    pred_a = all_actions[0][i].cpu().detach().numpy()
                    pred_a = (pred_a + 1)/2.0
                    a_mins5d = np.array([0.55, -0.035, 0.19, -90, 0.005])
                    a_maxs5d = np.array([0.63, 0.035, 0.25, 90, 0.05])
                    unnorm_pred_a = pred_a * (a_maxs5d - a_mins5d) + a_mins5d
                    print("Predicted action: ", unnorm_pred_a)
                    vis_grasp(state_unctr, goal_unctr, unnorm_pred_a, offset=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--temporal_agg', action='store_true')

    main(vars(parser.parse_args()))