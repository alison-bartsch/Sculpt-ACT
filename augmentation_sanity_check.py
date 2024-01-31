import torch
import numpy as np
import os
import argparse

from robot_utils import *
from os.path import join

import open3d as o3d

from imitate_clay_episodes import *
from action_geometry_utils import *

import IPython
e = IPython.embed

# TODO: given an index to a trajectory, sample a list of states, actions and next states
    # plot the state as blue
    # plot the next state as green 
    # plot the action as a red 

def visualize_pred_action_sequence(pred_actions, state_list, goal_list, ctr_path=None):
    # iterate through the predicted actions
    for i in range(pred_actions.shape[1]):
        # get the state/goal
        state = state_list[i]
        goal = goal_list[i]
        # unnormalize the action
        action = pred_actions[0][i]
        # print("\n\naction: ", action)
        action = unnormalize_a(action)  
        print("unnormalized a: ", action)

        # center action at origin of the point cloud
        pcl_center = np.array([0.6, 0, 0.25]) # [0.6, 0.0, 0.25]
        # pcl_center = state

        # pcl_center = np.load(ctr_path)
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
        # print("Action: ", action_cloud)
        ctr_action.points = o3d.utility.Vector3dVector(action_scaled[0:3].reshape(1,3))
        ctr_colors = np.tile(np.array([1, 0, 0]), (1,1))
        ctr_action.colors = o3d.utility.Vector3dVector(ctr_colors)
        o3d.visualization.draw_geometries([og_pcl, g_pcl, ctr_action, line_set, g1, g2, corners, g1_bbox])

if __name__ == '__main__':
    # no_ctr_path = '/home/alison/Clay_Data/Trajectory_Data/Aug_Dec14_Human_Demos/X/Trajectory877'
    no_ctr_path = '/home/alison/Clay_Data/Trajectory_Data/No_Aug_Dec14_Human_Demos/X/Trajectory4'
    gt_actions = []
    gt_states = []
    gt_next_states = []
    n_actions = 7
    for i in range(n_actions):
        action = np.load(no_ctr_path + '/action' + str(i) + '.npy')
        gt_actions.append(action)
        state = np.load(no_ctr_path + '/state' + str(i) + '.npy')
        gt_states.append(state)
        next_state = np.load(no_ctr_path + '/state' + str(i+1) + '.npy')
        gt_next_states.append(next_state)
    gt_actions = np.expand_dims(np.array(gt_actions), axis=0)
    print("GT ACTIONS SHAPE: ", gt_actions.shape)
    visualize_pred_action_sequence(gt_actions, gt_states, gt_next_states)