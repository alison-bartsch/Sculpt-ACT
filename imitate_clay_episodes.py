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
from embeddings.pointnet2 import *
from pointBERT.tools import builder
from pointBERT.utils.config import cfg_from_yaml_file

import json
from os.path import join

import open3d as o3d
from torch.optim.lr_scheduler import MultiStepLR

# from frankapy import FrankaArm

import IPython
e = IPython.embed

def main(args):
    set_seed(1)

    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # dataset_dir = '/home/alison/Clay_Data/Trajectory_Data/Embedded_X'
    # dataset_dir = '/home/alison/Clay_Data/Trajectory_Data/Aug24_Human_Demos/X'
    # dataset_dir = '/home/alison/Clay_Data/Trajectory_Data/No_Aug_Dec14_Human_Demos/X'




    dataset_dir = '/home/alison/Clay_Data/Trajectory_Data/Aug_Dec14_Human_Demos/X'
    # dataset_dir = '/home/alison/Clay_Data/Trajectory_Data/Aug_Jan24_Human_Demos_Stopping/X'
    num_episodes = 900 # 10 # 899 # 900
    episode_len = 9 # 6 # maximum episode lengths
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
        
        # if additional flag
            # load policy_config from file in ckpt_dir

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
        'action_pred': action_pred,
        # 'camera_names': camera_names,
        # 'real_robot': not is_sim
        'concat_goal': args['concat_goal'],
        'delta_goal': args['delta_goal'],
        'film_goal': args['film_goal'],
        'no_pos_embed': args['no_pos_embed'],
        'stopping_action': args['stopping_action'],
        'pointnet': args['pointnet'],
    }

    if is_eval:
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


def forward_pass_prev(data, policy, pointbert, encoder_head):
    qpos_data, state_data, action_data, is_pad = data
    qpos_data, state_data, action_data, is_pad = qpos_data.cuda(), state_data.cuda(), action_data.cuda(), is_pad.cuda()

    state_data = state_data.to(torch.float32)
    tokenized_states = pointbert(state_data)
    pcl_embed = encoder_head(tokenized_states)
    pcl_embed = torch.unsqueeze(pcl_embed, 1)

    return policy(qpos_data, pcl_embed, action_data, is_pad)

def forward_pass_no_pos(data, policy, pointbert, encoder_head):
    """
    Version of forward pass with goal conditioning.
    """
    goal_data, state_data, action_data, is_pad = data
    goal_data, state_data, action_data, is_pad = goal_data.cuda(), state_data.cuda(), action_data.cuda(), is_pad.cuda()

    state_data = state_data.to(torch.float32)
    tokenized_states = pointbert(state_data)
    pcl_embed = encoder_head(tokenized_states)
    pcl_embed = torch.unsqueeze(pcl_embed, 1)

    goal_data = goal_data.to(torch.float32)
    tokenized_goals = pointbert(goal_data)
    goal_embed = encoder_head(tokenized_goals)
    goal_embed = torch.unsqueeze(goal_embed, 1)

    return policy(goal_embed, pcl_embed, action_data, is_pad)

def forward_pass(data, policy, pointcloud_embed, encoder_head, concat_goal, delta_goal, film_goal, no_pos_embed, film_head=None):
    """
    Version of forward pass with goal conditioning.
    """
    goal_data, state_data, action_data, is_pad = data
    goal_data, state_data, action_data, is_pad = goal_data.cuda(), state_data.cuda(), action_data.cuda(), is_pad.cuda()

    if film_goal:
        state_data = state_data.to(torch.float32)
        tokenized_states = pointcloud_embed(state_data)
        goal_data = goal_data.to(torch.float32)
        tokenized_goals = pointcloud_embed(goal_data)

        pcl_embed = film_head(tokenized_states, tokenized_goals)
        pcl_embed = torch.unsqueeze(pcl_embed, 1)
        goal_embed = encoder_head(tokenized_goals)
        goal_embed = torch.unsqueeze(goal_embed, 1)
        return policy(goal_embed, pcl_embed, action_data, is_pad, concat_goal, delta_goal, no_pos_embed)

    else:
        state_data = state_data.to(torch.float32)
        tokenized_states = pointcloud_embed(state_data)
        pcl_embed = encoder_head(tokenized_states)
        pcl_embed = torch.unsqueeze(pcl_embed, 1)

        goal_data = goal_data.to(torch.float32)
        tokenized_goals = pointcloud_embed(goal_data)
        goal_embed = encoder_head(tokenized_goals)
        goal_embed = torch.unsqueeze(goal_embed, 1)
        return policy(goal_embed, pcl_embed, action_data, is_pad, concat_goal, delta_goal, no_pos_embed)    
    


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    encoder_frozen = config['encoder_frozen']
    pre_trained_encoder = config['pre_trained_encoder']
    action_pred = config['action_pred']
    pointnet = config['pointnet']

    # model variants
    concat_goal = config['concat_goal']
    delta_goal = config['delta_goal']
    film_goal = config['film_goal']
    if film_goal:
        concat_goal = True
    no_pos_embed = config['no_pos_embed']
    stopping_action = config['stopping_action'] # TODO: add this flag to influence the dataloader

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
                   'action_pred': action_pred,
                   'concat_goal': concat_goal,
                   'delta_goal': delta_goal,
                   'film_goal': film_goal,
                   'no_pos_embed': no_pos_embed,
                   'stopping_action': stopping_action
                }
    with open(join(ckpt_dir, 'params.json'), 'w') as f:
        json.dump(params_dict, f) 

    # save the policy config for experimental deployment
    with open(join(ckpt_dir, 'policy_config.json'), 'w') as f:
        json.dump(policy_config, f)

    set_seed(seed)

    # load point-BERT
    device = torch.device('cuda')
    if pointnet:
        pointcloud_embed = PointNet2().to(device) # pointnet (also rename pointbert to pointcloud_embed)
        encoder_head = PointNetProjection().to(device) # the mlp on top of the point cloud embedding for pointnet
        film_head = None
    else:
        pretrained_path = '/home/alison/Documents/GitHub/Point-BERT/embedding_experiments/exp1_statenextstate_contrastive' # exp24_new_dataset_pointbert_unfrozen'
            # exp24_new_dataset_pointbert_unfrozen
            # exp22_new_dataset_pointbert_unfrozen
            # exp1_statenextstate_contrastive
        if pre_trained_encoder:
            # enc_checkpoint = torch.load('pointBERT/encoder_weights/checkpoint', map_location=torch.device('cpu'))
            enc_checkpoint = torch.load(pretrained_path + '/checkpoint', map_location=torch.device('cpu'))
            encoder_head = enc_checkpoint['encoder_head'].to(device)
            if film_goal:
                encoded_dim = 768
                latent_dim = 512
                film_head = EncoderHeadFiLMPretrained(encoded_dim, latent_dim, encoder_head, encoded_dim).to(device)

            else:
                film_head = None
        else:
            encoded_dim = 768
            latent_dim = 512
            if film_goal:
                encoder_head = EncoderHeadFiLM(encoded_dim, latent_dim, encoded_dim).to(device)
            else:
                encoder_head = EncoderHead(encoded_dim, latent_dim).to(device)
                film_head = None

        config = cfg_from_yaml_file('pointBERT/cfgs/PointTransformer.yaml')
        model_config = config.model
        pointcloud_embed = builder.model_builder(model_config)
        # weights_path = 'pointBERT/point-BERT-weights/Point-BERT.pth'
        weights_path = pretrained_path + '/best_pointbert.pth'
        pointcloud_embed.load_model_from_ckpt(weights_path)
        pointcloud_embed.to(device)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    scheduler = MultiStepLR(optimizer,
                    milestones=[750, 1000],
                    gamma=0.5)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            encoder_head.eval()
            pointcloud_embed.eval()
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy, pointcloud_embed, encoder_head, concat_goal, delta_goal, film_goal, no_pos_embed, film_head)
                # forward_dict = forward_pass(data, policy, pointbert, encoder_head, pointbert_pos_embedding=False)
                # forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                print("Saving best encoder ckpts...")
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))

                # save point-BERT checkpoints
                # torch.save(pointbert.state_dict(), join(ckpt_dir, 'best_pointbert.pth'))

                # save encoder checkpoints
                if film_goal:
                    checkpoint = {'encoder_head': encoder_head, 
                                  'film_head': film_head}
                else:
                    checkpoint = {'encoder_head': encoder_head}
                torch.save(checkpoint, join(ckpt_dir, 'encoder_best_checkpoint'))

                torch.save({
                    'base_model' : pointcloud_embed.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'epoch' : epoch,
                    'metrics' : dict(),
                    'best_metrics' : dict(),
                    }, os.path.join(ckpt_dir, 'best_pointbert.pth'))

        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        if encoder_frozen:
            encoder_head.eval()
            pointcloud_embed.eval()
        else:
            encoder_head.train()
            pointcloud_embed.train()
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy, pointcloud_embed, encoder_head, concat_goal, delta_goal, film_goal, no_pos_embed, film_head)
            # forward_dict = forward_pass(data, policy, pointbert, encoder_head)
            # forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        scheduler.step()
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            # save policy checkpoints
            # ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            # torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    # ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    # torch.save(policy.state_dict(), ckpt_path)

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
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
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

    # modifications 
    parser.add_argument('--concat_goal', action='store', type=bool, default=True, help='Goal point cloud concatenation condition', required=False)
    parser.add_argument('--delta_goal', action='store', type=bool, default=False, help='Goal point cloud delta with state concatentation', required=False)
    parser.add_argument('--film_goal', action='store', type=bool, default=False, help='Goal point cloud FiLM condition', required=False)
    parser.add_argument('--pointnet', action='store', type=bool, default=False, help='Alternate point cloud embedding', required=False)
    parser.add_argument('--no_pos_embed', action='store', type=bool, default=False, help='No additional pos embedding (already in PointBERT embedding)', required=False)
    parser.add_argument('--stopping_action', action='store', type=bool, default=False, help='Add action dimension for stoping token', required=False)
    
    main(vars(parser.parse_args()))