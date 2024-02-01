import numpy as np
from draft_augmentation_utils import *

trajectory_path = '/home/alison/Clay_Data/Trajectory_Data/No_Aug_Dec14_Human_Demos/X/Trajectory9'
n_actions = 6
# iterate through n_action:
for i in range(n_actions):
    # load in a point cloud, goal and action
    state = np.load(trajectory_path + '/state' + str(i) + '.npy')
    goal_unnormalized = np.load(trajectory_path + '/goal.npy')
    action = np.load(trajectory_path + '/unnormalized_action' + str(i) + '.npy')
    center = np.load(trajectory_path + '/pcl_center' + str(i) + '.npy')
    # center goal point cloud
    goal = center_scale_cloud(goal_unnormalized)
    # unscale and center the state
    state_unnormalized = state * 0.1 + center

    # plot the original grasp action
    visualize_pred_action_sequence(action, state, goal) #, center)
    # vis_grasp_og_frame(state, goal, action, normalized=False, offset=False)

    for j in range(6):

        # augment the point cloud and grasp action
        rot = 60*j # [deg]
        
        state_aug, action_aug, goal_aug = augment_state_action(state_unnormalized, center, action, goal_unnormalized, rot)

        # plot the augmented grasp action
        visualize_pred_action_sequence(action_aug, state_aug, goal_aug) #, center)
        # vis_grasp_og_frame(state_aug, goal_aug, action_aug, normalized=False, offset=False)

    assert False

