import numpy as np
import torch
import os
import h5py
import math
import copy
import time
import open3d as o3d
from os.path import exists
from torch.utils.data import TensorDataset, DataLoader
# from visualize_predicted_sequence import visualize_grasp

import IPython
e = IPython.embed

class ClayDataset(torch.utils.data.Dataset):
    def __init__(self, episode_idxs, dataset_dir, n_datapoints, n_raw_trajectories, center_action=False, stopping_token=False):
        """
        The Dataloader for the clay sculpting dataset at the Trajectory level (compatible with ACT and Diffusion Policy). 

        :param episode_idxs: list of indices of the episodes to load
        :param dataset_dir: directory where the dataset is stored
        :param n_datapoints: number of datapoints (i.e. desired number of final trajectories after augmentation)
        :param n_raw_trajectories: number of raw trajectories in the dataset
        :param center_action: whether to center the action before normalizing
        :param stopping_token: whether to add a stopping token to the action [not currently implemented TODO]
        """
        super(ClayDataset).__init__()
        self.dataset_dir = dataset_dir
        self.episode_idxs = episode_idxs
        self.max_len = 9 # maximum number of actions for X trajectory
        self.action_shape = (self.max_len, 5)
        self.n_datapoints = n_datapoints
        self.n_raw_trajectories = n_raw_trajectories
        self.center_action = center_action
        self.stopping_token = stopping_token

        # determine the number of datapoints per trajectory - needs to be a round number
        self.n_datapoints_per_trajectory = self.n_datapoints / self.n_raw_trajectories
        if not self.n_datapoints_per_trajectory.is_integer():
            raise ValueError('The number of datapoints per trajectory needs to be a round number, please input a valid number of datapoints given the number of raw trajectories')

        # deterime the augmentation interval
        self.aug_step = 360 / self.n_datapoints_per_trajectory

    def _center_pcl(self, pcl, center):
        centered_pcl = pcl - center
        centered_pcl = centered_pcl * 10
        return centered_pcl

    def _center_normalize_action(self, action, ctr):
        # center the action
        new_action = np.zeros(5)
        new_action[0:3] = action[0:3] - ctr
        new_action[3:5] = action[3:5]
        # normalize centered action
        mins = np.array([-0.15, -0.15, -0.05, -90, 0.005])
        maxs = np.array([0.15, 0.15, 0.05, 90, 0.05])
        norm_action = np.zeros(6)
        norm_action[0:5] = (action[0:5] - mins) / (maxs - mins)
        return norm_action

    def _normalize_action(self, action):
        a_mins5d = np.array([0.55, -0.035, 0.19, -90, 0.005])
        a_maxs5d = np.array([0.63, 0.035, 0.25, 90, 0.05])
        norm_action = (action - a_mins5d) / (a_maxs5d - a_mins5d)
        norm_action = norm_action * 2 - 1 # set to [-1, 1]
        return norm_action

    def _rotate_pcl(self, state, center, rot):
        '''
        Faster implementation of rotation augmentation to fix slow down issue
        '''
        state = state - center
        R = o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz((0, 0, math.radians(rot)))
        state = R @ state.T
        pcl_aug = state.T + center
        return pcl_aug
    
    def _wrap_rz(self, original_rz):
        wrapped_rz = (original_rz + 90) % 180 - 90
        return wrapped_rz

    def _rotate_action(self, action, center, rot):
        unit_circle_og_grasp = (action[0] - center[0], action[1] - center[1])
        rot_original = math.atan2(unit_circle_og_grasp[1], unit_circle_og_grasp[0])
        unit_circle_radius = math.sqrt(unit_circle_og_grasp[0]**2 + unit_circle_og_grasp[1]**2)
        rot_new =  rot_original + rot
        new_unit_circle_grasp = (-unit_circle_radius*math.cos(math.radians(rot_new)), -unit_circle_radius*math.sin(math.radians(rot_new)))
        
        new_global_grasp = (center[0] + new_unit_circle_grasp[0], center[1] + new_unit_circle_grasp[1])
        x = new_global_grasp[0]
        y = new_global_grasp[1]
        rz = action[3] + rot
        rz = self._wrap_rz(rz)
        action_aug = np.array([x, y, action[2], rz, action[4]])
        return action_aug
    
    def __len__(self):
        """
        Return the number of episodes in the dataset (i.e. the number of actions in the trajectory folder)
        """
        return len(self.episode_idxs)

    def __getitem__(self, index):
        sample_full_episode = True # hardcode

        # built in ACT functionality to determine idx randomness
        idx = self.episode_idxs[index]
        # determine which raw_trajectory to index
        raw_traj_idx = int(idx // self.n_datapoints_per_trajectory)
        # determine the rotation augmentation to apply
        aug_rot = (idx % self.n_datapoints_per_trajectory) * self.aug_step
        traj_path = self.dataset_dir + '/Trajectory' + str(raw_traj_idx)

        states = []
        actions = []
        j = 0

        # TODO: check for starting from state = 3

        # iterate loading in the actions as long as the next state point cloud exists
        # while exists(traj_path + '/state' + str(j) + '.npy'): # TODO: FOR JAN DATASET
        while exists(traj_path + '/unnormalized_state' + str(j) + '.npy'):
            # load the center
            ctr = np.load(traj_path + '/pcl_center' + str(j) + '.npy')
            # load the uncentered state
            # s = np.load(traj_path + '/state' + str(j) + '.npy') # TODO: FOR JAN DATASET
            s = np.load(traj_path + '/unnormalized_state' + str(j) + '.npy')
            # apply state rotation
            s_rot = self._rotate_pcl(s, ctr, aug_rot)
            # center and scale state
            s_rot_scaled = self._center_pcl(s_rot, ctr)
            # append to state list
            states.append(s_rot_scaled)

            if j != 0:
                # load unnormalized action
                # a = np.load(traj_path + '/action' + str(j-1) + '.npy') # TODO: FOR JAN DATASET
                a = np.load(traj_path + '/unnormalized_action' + str(j-1) + '.npy')
                # apply action rotation
                a_rot = self._rotate_action(a, ctr, aug_rot)
                # normalize action (can choose between normalization strategies)
                if self.center_action:
                    # center and normalize action
                    a_scaled = self._center_normalize_action(a_rot, ctr)
                else:
                    # just normalize action
                    a_scaled = self._normalize_action(a_rot)
                
                # add stopping token if necessary
                if self.stopping_token:
                    # set the stopping token
                    # action[5] = int(stopping == True)
                    pass

                actions.append(a_scaled)
            j+=1

        episode_len = len(actions)
        start_ts = np.random.choice(episode_len)
        state = states[start_ts]

        # TODO: MOVE ROTATION AUGMENTATION OF STATE AND GOAL OUTSIDE OF LOOP!!!!!!
        
        # load uncentered goal
        g = np.load(traj_path + '/goal.npy')
        # apply goal rotation
        g_rot = self._rotate_pcl(g, ctr, aug_rot)
        # center and scale goal
        goal = self._center_pcl(g_rot, ctr)

        action = actions[start_ts:]
        action = np.stack(action, axis=0)

        # get obs_pos as previous action
        if start_ts != 0:
            obs_pos = actions[start_ts-1]
        else:
            obs_pos = self._normalize_action(np.array([0.6, 0.0, 0.25, 0.0, 0.05]))

        action_len = episode_len - start_ts

        padded_action = np.zeros(self.action_shape, dtype=np.float32)

        padded_action[:action_len] = action
        is_pad = np.zeros(self.max_len)
        is_pad[action_len:] = 1

        # construct observations
        state_data = torch.from_numpy(state)
        goal_data = torch.from_numpy(goal).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        obs_pos = torch.from_numpy(obs_pos).float()

        return goal_data, state_data, obs_pos, action_data, is_pad


# class ClayDataset(torch.utils.data.Dataset):
#     def __init__(self, episode_idxs, dataset_dir, action_pred): #, visualize_grasp):
#         """
#         NOTE: The point clouds and actions are already normalized.
#         """
#         super(ClayDataset).__init__()
#         self.dataset_dir = dataset_dir
#         self.episode_idxs = episode_idxs
#         self.max_len = 9 # 6 # maximum number of actions for X trajectory
#         # self.action_shape = (self.max_len, 6)
#         self.action_shape = (self.max_len, 5)
#         self.action_pred = action_pred
        

#     def _center_pcl(self, state, pcl, ctr):
#         pass

#     def _center_normalize_action(self, action, ctr, std):
#         pass

#     def _rotation_augmentation(self, state_list, action_list, goal):
#         pass
#         # TODO: implement rotation augmentation to reduce dataset storage size
    
#         # ACTION NEEDS TO BE UNNORMALIZED AND UNCENTERED!!! THIS WILL OCCUR AFTER THE AUGMENTATION
#         # IS APPLIED!!!!
    
#         # STATE AND GOALS NEED TO BE UNNORMALIZED 
    
#     def __len__(self):
#         """
#         Return the number of episodes in the dataset (i.e. the number of actions in the trajectory folder)
#         """
#         return len(self.episode_idxs)

#     def __getitem__(self, index):
#         sample_full_episode = True # hardcode

#         idx = self.episode_idxs[index]
#         traj_path = self.dataset_dir + '/Trajectory' + str(idx)

#         states = []
#         actions = []
#         j = 0

#         # while exists(traj_path + '/state' + str(j) + '.npy'):
#         while exists(traj_path + '/pointcloud' + str(j) + '.npy'):
#             # if we are predicting grasp action, state is pcl and action is grasp
#             if self.action_pred:
#                 # s = np.load(traj_path + '/state' + str(j) + '.npy')
#                 s = np.load(traj_path + '/pointcloud' + str(j) + '.npy')
#             # if we are predicting intermediate states, state is grasp and action is intermediate state
#             else:
#                 s = np.load(traj_path + '/action' + str(j) + '.npy')
#             states.append(s)

#             if j != 0:
#                 if self.action_pred:
#                     a = np.load(traj_path + '/action' + str(j-1) + '.npy')
#                     # a = np.load(traj_path + '/action_normalized_ctr' + str(j-1) + '.npy')
#                 else:
#                     # a = np.load(traj_path + '/state' + str(j) + '.npy')
#                     a = np.load(traj_path + '/pointcloud' + str(j) + '.npy')
#                 actions.append(a)
#             j+=1

#         episode_len = len(actions)
#         # print("\n\nEpisode len: ", episode_len)
#         start_ts = np.random.choice(episode_len)
#         # print("start ts: ", start_ts)
#         state = states[start_ts]

#         # # visualize state
#         # state_o3d = o3d.geometry.PointCloud()
#         # state_o3d.points = o3d.utility.Vector3dVector(state)
#         # state_o3d_colors = np.tile(np.array([0,1,0]), (len(state),1))
#         # state_o3d.colors = o3d.utility.Vector3dVector(state_o3d_colors)
#         # # o3d.visualization.draw_geometries([state_o3d])

#         # # visualize remaining state trajectory
#         # remaining_states = states[start_ts:]
#         # for i in range(len(remaining_states)):
#         #     pcl = remaining_states[i]
#         #     pcl_o3d = o3d.geometry.PointCloud()
#         #     pcl_o3d.points = o3d.utility.Vector3dVector(pcl)
#         #     pcl_o3d_colors = np.tile(np.array([0,0,1]), (len(pcl),1))
#         #     pcl_o3d.colors = o3d.utility.Vector3dVector(pcl_o3d_colors)
#         #     # o3d.visualization.draw_geometries([state_o3d, pcl_o3d])

#         #     if i != 0:
#         #         self.visualize_grasp(remaining_states[i-1], remaining_states[i], actions[i-1])


#         # goal = states[-1]
#         goal = np.load(traj_path + '/goal.npy')

#         # # visualize final state (goal)
#         # goal_o3d = o3d.geometry.PointCloud()
#         # goal_o3d.points = o3d.utility.Vector3dVector(goal)
#         # goal_o3d_colors = np.tile(np.array([1,0,0]), (len(goal),1))
#         # goal_o3d.colors = o3d.utility.Vector3dVector(goal_o3d_colors)
#         # o3d.visualization.draw_geometries([pcl_o3d, goal_o3d])

#         action = actions[start_ts:]
#         action = np.stack(action, axis=0)
#         # print("\naction: ", action)
#         action_len = episode_len - start_ts

#         padded_action = np.zeros(self.action_shape, dtype=np.float32)

#         padded_action[:action_len] = action
#         is_pad = np.zeros(self.max_len)
#         is_pad[action_len:] = 1

#         # construct observations
#         state_data = torch.from_numpy(state)
#         goal_data = torch.from_numpy(goal).float()
#         action_data = torch.from_numpy(padded_action).float()
#         is_pad = torch.from_numpy(is_pad).bool()

#         return goal_data, state_data, action_data, is_pad
class ClayDatasetPrev(torch.utils.data.Dataset):
    def __init__(self, episode_idxs, dataset_dir, action_pred):
        """
        NOTE: The point clouds and actions are already normalized.
        """
        super(ClayDataset).__init__()
        self.dataset_dir = dataset_dir
        self.episode_idxs = episode_idxs
        self.max_len = 6 # maximum number of actions for X trajectory
        self.action_shape = (self.max_len, 5)
        self.action_pred = action_pred
    
    def __len__(self):
        """
        Return the number of episodes in the dataset (i.e. the number of actions in the trajectory folder)
        """
        return len(self.episode_idxs)

    def __getitem__(self, index):
        sample_full_episode = True # hardcode

        idx = self.episode_idxs[index]
        traj_path = self.dataset_dir + '/Trajectory' + str(idx)

        states = []
        actions = []
        j = 0

        while exists(traj_path + '/state' + str(j) + '.npy'):
            # if we are predicting grasp action, state is pcl and action is grasp
            if self.action_pred:
                s = np.load(traj_path + '/state' + str(j) + '.npy')
            # if we are predicting intermediate states, state is grasp and action is intermediate state
            else:
                s = np.load(traj_path + '/action' + str(j) + '.npy')
            states.append(s)

            if j != 0:
                if self.action_pred:
                    a = np.load(traj_path + '/action' + str(j-1) + '.npy')
                else:
                    a = np.load(traj_path + '/state' + str(j) + '.npy')
                actions.append(a)
            j+=1

        episode_len = len(actions)
        start_ts = np.random.choice(episode_len)
        state = states[start_ts]
        action = actions[start_ts:]
        action = np.stack(action, axis=0)
        action_len = episode_len - start_ts

        qpos = np.ones(self.action_shape[1])
        padded_action = np.zeros(self.action_shape, dtype=np.float32)

        padded_action[:action_len] = action
        is_pad = np.zeros(self.max_len)
        is_pad[action_len:] = 1

        # construct observations
        state_data = torch.from_numpy(state)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        return qpos_data, state_data, action_data, is_pad


class ClayDatasetEmbedded(torch.utils.data.Dataset):
    def __init__(self, episode_idxs, dataset_dir):
        """
        NOTE: The point clouds and actions are already normalized.
        """
        super(ClayDatasetEmbedded).__init__()
        self.dataset_dir = dataset_dir
        self.episode_idxs = episode_idxs
        self.max_len = 6 # maximum number of actions for X trajectory
        self.action_shape = (self.max_len, 5)
    
    def __len__(self):
        """
        Return the number of episodes in the dataset (i.e. the number of actions in the trajectory folder)
        """
        return len(self.episode_idxs)

    def __getitem__(self, index):
        sample_full_episode = True # hardcode

        idx = self.episode_idxs[index]
        traj_path = self.dataset_dir + '/Trajectory' + str(idx)
        # print("\nTrajectory path: ", traj_path)

        states = []
        actions = []
        j = 0

        # print("\n\n\nPath: ", traj_path + '/s_embed' + str(j))

        # load the entire trajectory
        # while exists(traj_path + '/state' + str(j) + '.npy'):
        while exists(traj_path + '/s_embed' + str(j) + '.npy'):
            # print("Path exists!")
            s = np.load(traj_path + '/s_embed' + str(j) + '.npy')
            # s = torch.from_numpy(s).float()
            states.append(s)

            if j != 0:
                a = np.load(traj_path + '/action' + str(j-1) + '.npy')
                # a = torch.from_numpy(a).float()
                actions.append(a)
                # print("action appended")
            j+=1

        # print("\n\nActions: ", actions)
        episode_len = len(actions)
        # print("\n\nEpisode len: ", episode_len)
        # print("\n\n\n\n\n\n")
        start_ts = np.random.choice(episode_len)
        # print("\nStart ts: ", start_ts)
        state = states[start_ts]
        action = actions[start_ts:]
        # print("\nlen(action): ", len(action))
        action = np.stack(action, axis=0)
        action_len = episode_len - start_ts

        qpos = np.ones(self.action_shape[1])
        padded_action = np.zeros(self.action_shape, dtype=np.float32)
        # print("\nPadded Action Shape: ", padded_action.shape)
        # print("Action shape: ", action.shape)

        padded_action[:action_len] = action
        is_pad = np.zeros(self.max_len)
        is_pad[action_len:] = 1

        # construct observations
        state_data = torch.from_numpy(state)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        return qpos_data, state_data, action_data, is_pad

        # iterate through length
            # import states and actions
        
        # return list of states and list of actions
        # final state is the goal state

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            # print("\nOriginal Action Shape: ", original_action_shape)
            episode_len = original_action_shape[0]
            # print("\nOriginal Episode Length: ", episode_len)
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        # print("Individual Action Sequence Length: ", action.shape)

        self.is_sim = is_sim
        # print("\nOriginal Action Shape: ", original_action_shape)
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1
        # print("\nis pad for action: ", is_pad)
        # print("Sum is pad ", np.sum(is_pad))
        # print("Episode len ", episode_len)
        # print("action len: ", action_len)

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # print("\nqpos data shape: ", qpos_data.shape)
        # print("qpos: ", qpos_data)

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats

def load_clay_data(dataset_dir, num_episodes, batch_size_train, batch_size_val, action_pred, n_datapoints, n_raw_trajectories): #, visualize_grasp):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # construct dataset and dataloader
    train_dataset = ClayDataset(train_indices, dataset_dir, n_datapoints, n_raw_trajectories)
    val_dataset = ClayDataset(train_indices, dataset_dir, n_datapoints, n_raw_trajectories)
    # train_dataset = ClayDataset(train_indices, dataset_dir, action_pred) #, visualize_grasp)
    # val_dataset = ClayDataset(val_indices, dataset_dir, action_pred) #, visualize_grasp)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    assert False
    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
