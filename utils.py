import numpy as np
import torch
import os
import h5py
import open3d as o3d
from os.path import exists
from torch.utils.data import TensorDataset, DataLoader
# from visualize_predicted_sequence import visualize_grasp

import IPython
e = IPython.embed

class ClayDataset(torch.utils.data.Dataset):
    def __init__(self, episode_idxs, dataset_dir, action_pred): #, visualize_grasp):
        """
        NOTE: The point clouds and actions are already normalized.
        """
        super(ClayDataset).__init__()
        self.dataset_dir = dataset_dir
        self.episode_idxs = episode_idxs
        self.max_len = 6 # maximum number of actions for X trajectory
        self.action_shape = (self.max_len, 5)
        self.action_pred = action_pred
        # self.visualize_grasp = visualize_grasp
    
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
        # print("\n\nEpisode len: ", episode_len)
        start_ts = np.random.choice(episode_len)
        # print("start ts: ", start_ts)
        state = states[start_ts]

        # # visualize state
        # state_o3d = o3d.geometry.PointCloud()
        # state_o3d.points = o3d.utility.Vector3dVector(state)
        # state_o3d_colors = np.tile(np.array([0,1,0]), (len(state),1))
        # state_o3d.colors = o3d.utility.Vector3dVector(state_o3d_colors)
        # # o3d.visualization.draw_geometries([state_o3d])

        # # visualize remaining state trajectory
        # remaining_states = states[start_ts:]
        # for i in range(len(remaining_states)):
        #     pcl = remaining_states[i]
        #     pcl_o3d = o3d.geometry.PointCloud()
        #     pcl_o3d.points = o3d.utility.Vector3dVector(pcl)
        #     pcl_o3d_colors = np.tile(np.array([0,0,1]), (len(pcl),1))
        #     pcl_o3d.colors = o3d.utility.Vector3dVector(pcl_o3d_colors)
        #     # o3d.visualization.draw_geometries([state_o3d, pcl_o3d])

        #     if i != 0:
        #         self.visualize_grasp(remaining_states[i-1], remaining_states[i], actions[i-1])


        goal = states[-1]

        # # visualize final state (goal)
        # goal_o3d = o3d.geometry.PointCloud()
        # goal_o3d.points = o3d.utility.Vector3dVector(goal)
        # goal_o3d_colors = np.tile(np.array([1,0,0]), (len(goal),1))
        # goal_o3d.colors = o3d.utility.Vector3dVector(goal_o3d_colors)
        # o3d.visualization.draw_geometries([pcl_o3d, goal_o3d])

        action = actions[start_ts:]
        action = np.stack(action, axis=0)
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

        # print("Goal Data: ", goal_data.shape)
        # print("State data: ", state_data.shape)
        # print("Action data: ", action_data)
        # print("Is pad: ", is_pad)

        return goal_data, state_data, action_data, is_pad
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

def load_clay_data(dataset_dir, num_episodes, batch_size_train, batch_size_val, action_pred): #, visualize_grasp):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # construct dataset and dataloader
    train_dataset = ClayDataset(train_indices, dataset_dir, action_pred) #, visualize_grasp)
    val_dataset = ClayDataset(val_indices, dataset_dir, action_pred) #, visualize_grasp)
    # train_dataset = ClayDatasetPrev(train_indices, dataset_dir, action_pred)
    # val_dataset = ClayDatasetPrev(val_indices, dataset_dir, action_pred)
    # train_dataset = ClayDatasetEmbedded(train_indices, dataset_dir)
    # val_dataset = ClayDatasetEmbedded(val_indices, dataset_dir)
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
