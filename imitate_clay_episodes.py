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
from dynamics.dynamics_model import EncoderHead
from pointBERT.tools import builder
from pointBERT.utils.config import cfg_from_yaml_file

import json
from os.path import join

from frankapy import FrankaArm

import IPython
e = IPython.embed

def main(args):
    set_seed(1)

    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    # task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # dataset_dir = '/home/alison/Clay_Data/Trajectory_Data/Embedded_X'
    dataset_dir = '/home/alison/Clay_Data/Trajectory_Data/Aug24_Human_Demos/X'
    num_episodes = 899 # 900
    episode_len = 6 # maximum episode lengths
    encoder_frozen = False
    pre_trained_encoder = False
    action_pred = True

    # fixed parameters
    state_dim = 5 #14
    lr_backbone = 1e-5
    backbone = 'resnet18' # TODO: FIX THIS!
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                        #  'camera_names': camera_names,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'policy_config': policy_config,
        # 'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'encoder_frozen': encoder_frozen,
        'pre_trained_encoder': pre_trained_encoder,
        'action_pred': action_pred
        # 'camera_names': camera_names,
        # 'real_robot': not is_sim
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = clay_eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    # train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, batch_size_train, batch_size_val)
    train_dataloader, val_dataloader = load_clay_data(dataset_dir, num_episodes, batch_size_train, batch_size_val, action_pred)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        # print("\nPolicy Config: ", policy_config)
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def clay_eval_bc(config, ckpt_name, save_episode=True):
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
    enc_checkpoint = torch.load('pointBERT/encoder_weights/checkpoint', map_location=torch.device('cpu'))
    encoder_head = enc_checkpoint['encoder_head'].to(device)
    config = cfg_from_yaml_file('pointBERT/cfgs/PointTransformer.yaml')
    model_config = config.model
    pointbert = builder.model_builder(model_config)
    weights_path = 'pointBERT/point-BERT-weights/Point-BERT.pth'
    pointbert.load_model_from_ckpt(weights_path)
    pointbert.to(device)

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks
    num_rollouts = 5
    for rollout_id in range(num_rollouts):
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

            for t in range(max_timesteps):
                # get the current state (point cloud)
                _, _, pc2, _ = cam2._get_next_frame()
                _, _, pc3, _ = cam3._get_next_frame()
                _, _, pc4, _ = cam4._get_next_frame()
                _, _, pc5, _ = cam5._get_next_frame()
                pointcloud = pcl_vis.fuse_point_clouds(pc2, pc3, pc4, pc5, vis=True)

                # pass the point cloud through Point-BERT to get the latent representation
                state = torch.from_numpy(pointcloud).to(torch.float32)
                states = torch.unsqueeze(state, 0).to(device)
                print("states shape: ", states.shape)

                # pass through Point-BERT
                tokenized_states = pointbert(states)
                pcl_embed = encoder_head(tokenized_states)
                pcl_embed = torch.unsqueeze(pcl_embed, 1)
                print("pcl_embed shape: ", pcl_embed.shape)

                # set qpos to ones
                qpos = np.ones(5)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

                ### query policy
                if t % query_frequency == 0:
                    all_actions = policy(qpos, pcl_embed)
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
                fa.open_gripper()

                # move to observation pose
                pose.translation = observation_pose
                fa.goto_pose(pose)

        # wait for ENTER key for the clay to be reset
        input("Press ENTER to continue when clay has been reset...")


def forward_pass(data, policy, pointbert, encoder_head):
    qpos_data, state_data, action_data, is_pad = data
    qpos_data, state_data, action_data, is_pad = qpos_data.cuda(), state_data.cuda(), action_data.cuda(), is_pad.cuda()

    state_data = state_data.to(torch.float32)
    tokenized_states = pointbert(state_data)
    pcl_embed = encoder_head(tokenized_states)
    pcl_embed = torch.unsqueeze(pcl_embed, 1)

    return policy(qpos_data, pcl_embed, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    encoder_frozen = config['encoder_frozen']
    pre_trained_encoder = config['pre_trained_encoder']
    action_pred = config['action_pred']

    # create dictionary of all the key training info
    params_dict = {'num_epochs': num_epochs,
                   'seed': seed,
                   'policy_class': policy_class,
                   'lr': policy_config['lr'],
                   'kl_weight': policy_config['kl_weight'],
                   'chunk_size': policy_config['num_queries'],
                   'hidden_dim': policy_config['hidden_dim'],
                   'encoder_frozen': encoder_frozen,
                   'pre_trained_encoder': pre_trained_encoder,
                   'action_pred': action_pred
                }
    with open(join(ckpt_dir, 'params.json'), 'w') as f:
        json.dump(params_dict, f) 

    set_seed(seed)

    # load point-BERT
    device = torch.device('cuda')
    if pre_trained_encoder:
        enc_checkpoint = torch.load('pointBERT/encoder_weights/checkpoint', map_location=torch.device('cpu'))
        encoder_head = enc_checkpoint['encoder_head'].to(device)
    else:
        encoded_dim = 768
        latent_dim = 512
        encoder_head = EncoderHead(encoded_dim, latent_dim).to(device)
    config = cfg_from_yaml_file('pointBERT/cfgs/PointTransformer.yaml')
    model_config = config.model
    pointbert = builder.model_builder(model_config)
    weights_path = 'pointBERT/point-BERT-weights/Point-BERT.pth'
    pointbert.load_model_from_ckpt(weights_path)
    pointbert.to(device)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            encoder_head.eval()
            pointbert.eval()
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy, pointbert, encoder_head)
                # forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        if encoder_frozen:
            encoder_head.eval()
            pointbert.eval()
        else:
            encoder_head.train()
            pointbert.train()
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy, pointbert, encoder_head)
            # forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save the best val loss to a txt file
    with open(ckpt_dir + '/best_val_loss.txt', 'w') as f:
        string = str(best_epoch) + ':   ' + str(min_val_loss)
        f.write(string)

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    # parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    # args = parser.parse_args()
    # print("args: ", args)

    # args_dict = vars(args)
    # print("vars: ", args_dict)
    # main(args_dict)
    # # assert False
    
    main(vars(parser.parse_args()))
    # main(vars(args))
