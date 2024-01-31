import numpy as np
from draft_augmentation_utils import *

trajectory_path = '/home/alison/Clay_Data/Trajectory_Data/No_Aug_Dec14_Human_Demos/X/Trajectory5'
n_actions = 6
# iterate through n_action:
for i in range(n_actions):
    # load in a point cloud, goal and action
    state = np.load(trajectory_path + '/state' + str(i) + '.npy')
    goal = np.load(trajectory_path + '/goal.npy')
    action = np.load(trajectory_path + '/unnormalized_action' + str(i) + '.npy')
    center = np.load(trajectory_path + '/pcl_center' + str(i) + '.npy')
    # center goal point cloud
    goal = center_scale_cloud(goal)

    # plot the original grasp action
    visualize_pred_action_sequence(action, state, goal)

    # augment the point cloud and grasp action
    rot = 200 # [deg]
    
    state_aug, action_aug, goal_aug = augment_state_action(state, center, action, goal, rot)

    # plot the augmented grasp action
    visualize_pred_action_sequence(action_aug, state_aug, goal_aug)

