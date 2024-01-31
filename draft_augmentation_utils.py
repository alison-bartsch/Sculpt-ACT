import torch
import numpy as np
import os
import copy
import argparse
import math

from robot_utils import *
from os.path import join

import open3d as o3d

from imitate_clay_episodes import *
from action_geometry_utils import *

import IPython
e = IPython.embed

def center_scale_cloud(state):
    # center and scale cloud
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(state)
    ctr = pcl.get_center()
    centered_pcl = state - ctr
    centered_pcl = centered_pcl * 10
    return centered_pcl

def wrap_rz(original_rz):
    """
    We want rz to be between -90 to 90, so wrap around if outside these bounds due to symmetrical gripper.
    """
    wrapped_rz = (original_rz + 90) % 180 - 90
    return wrapped_rz

def augment_state_action(state, center, action, goal, rot):
    """
    state: unnormalized point cloud

    """
    # apply rotation augmentation to unnormalized point cloud
    unnorm_pointcloud = o3d.geometry.PointCloud()
    unnorm_pointcloud.points = o3d.utility.Vector3dVector(state)
    unscale_pcl = copy.deepcopy(unnorm_pointcloud)
    unscale_pcl.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 1]), (state.shape[0],1)))
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    R = mesh.get_rotation_matrix_from_xyz((0, 0, math.radians(rot)))
    unscale_pcl.rotate(R, center=center)
    pcl_aug = np.asarray(unscale_pcl.points)

    # apply rotation augmentation to unnormalized goal point cloud
    unnorm_goal = o3d.geometry.PointCloud()
    unnorm_goal.points = o3d.utility.Vector3dVector(goal)
    unscale_goal = copy.deepcopy(unnorm_goal)
    unscale_goal.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 1]), (goal.shape[0],1)))
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    R = mesh.get_rotation_matrix_from_xyz((0, 0, math.radians(rot)))
    unscale_goal.rotate(R, center=center)
    goal_aug = np.asarray(unscale_goal.points)

    # apply rotation augmentation to unnormalized action
    # print("\naction: ", action)
    # unit_circle_og_grasp = (action[0] - center[0], action[1] - center[1])
    # # unit_circle_og_grasp = (center[0] - action[0], center[1] - action[1])
    # print("unit_circle_og_grasp: ", unit_circle_og_grasp)
    # unit_circle_radius = math.sqrt(unit_circle_og_grasp[0]**2 + unit_circle_og_grasp[1]**2)
    # print("unit_circle_radius: ", unit_circle_radius)
    # rot_original = math.atan2(unit_circle_og_grasp[1], unit_circle_og_grasp[0])
    # print("rot_original: ", math.degrees(rot_original))
    # print("rz original: ", 180 - action[3])
    # rot_new =  rot_original + rot
    # print("rot_new: ", math.degrees(rot_new))
    # new_unit_circle_grasp = (unit_circle_radius*math.cos(math.radians(rot_new)), unit_circle_radius*math.sin(math.radians(rot_new)))
    # print("new_unit_circle_grasp: ", new_unit_circle_grasp)

    # # TODO: new_unit_circle_grasp is not correct - double check the calculation
    # print("Test1: ", (unit_circle_radius*math.cos(math.radians(-rot_new)), unit_circle_radius*math.sin(math.radians(-rot_new))))
    # print("Test1: ", (-unit_circle_radius*math.cos(math.radians(-rot_new)), -unit_circle_radius*math.sin(math.radians(-rot_new))))

    # # ORIGINAL ---- NOT WORKING PROPERLY
    # new_global_grasp = (center[0] + new_unit_circle_grasp[0], center[1] + new_unit_circle_grasp[1])

    # TODO: NEED AN ANGLE REPRESENTATION THAT MATCHES unit_circle_og_grasp with unit_circle_recon_grasp

    unit_circle_og_grasp = (action[0] - center[0], action[1] - center[1])
    unit_circle_radius = math.sqrt(unit_circle_og_grasp[0]**2 + unit_circle_og_grasp[1]**2)
    # rot_original = action[3] # + 90 # because we set zero arbitrarily to be 90 in robot frame
    # print("\norignal rz: ", rot_original)
    rot_original = math.degrees(math.asin(unit_circle_og_grasp[1]/unit_circle_radius))
    # print("\n\nrecon rz: ", rot_original)
    
    
        
    # unit_circle_recon_grasp = (-unit_circle_radius*math.cos(math.radians(rot_original)), unit_circle_radius*math.sin(math.radians(rot_original)))
    unit_circle_recon_grasp = (-unit_circle_radius*math.sin(math.radians(rot_original + 90)), -unit_circle_radius*math.cos(math.radians(rot_original + 90)))
    
    rot_new =  rot_original + 90 + rot
    # rot_new = rot_original + rot 
    # rot_new = rot_original - 90 - rot
    print("\n\n\nrot_new: ", rot_new)

    # FOR ROT_NEW + ROT_ORIGINAL + ROT
    # OBSERVATION FROM ROT = 0
        # if -90 < nrot_new < 90:
            # -cos, sin
            # fixes sanity check
            # corrects everything
    # OBSERVATION FROM ROT = 50
        # if -90 < rot_new < 90:
            # -cos, sin
            # fixes sanity check, but breaks final position
        # if rot_new == 108:
            # cos, -sin fixes everything
    
    # FOR ROT_NEW + ROT_ORIGINAL + 90 + ROT
    # OBSERVATION FROM ROT = 50
        # if rot_new > 180:
            # sin, cos
        # if rot_new < 180:
            # 
    
    # NOTE: we need to modify the rotation by 90 degrees due to how we set up the robot coordinate frame
    
    print("unit_circle_og_grasp: ", unit_circle_og_grasp)
    print("sanity check: ", unit_circle_recon_grasp)
    print("center: ", center)
    # new_unit_circle_grasp = (unit_circle_radius*math.cos(math.radians(rot_new)), unit_circle_radius*math.sin(math.radians(rot_new)))
    if rot_new > 180:
        # WORKING CORRECTLY FOR TRAJ 5 ROT AUGMENTATION = 50
        new_unit_circle_grasp = (unit_circle_radius*math.sin(math.radians(rot_new)), unit_circle_radius*math.cos(math.radians(rot_new)))
    elif rot_new > 90:
        # # WORKING CORRECTLY FOR TRAJ 5 NO ROT AUGMENTATION
        new_unit_circle_grasp = (-unit_circle_radius*math.sin(math.radians(rot_new)), unit_circle_radius*math.cos(math.radians(rot_new)))

        # new_unit_circle_grasp = (unit_circle_radius*math.cos(math.radians(rot_new)), -unit_circle_radius*math.sin(math.radians(rot_new)))
    elif rot_new > 0:
        # WORKING CORRECTLY FOR TRAJ 5 NO ROT AUGMENTATION
        # new_unit_circle_grasp = (-unit_circle_radius*math.sin(math.radians(rot_new)), -unit_circle_radius*math.cos(math.radians(rot_new)))

        new_unit_circle_grasp = (unit_circle_radius*math.sin(math.radians(rot_new)), -unit_circle_radius*math.cos(math.radians(rot_new)))
    else:
        new_unit_circle_grasp = (-unit_circle_radius*math.sin(math.radians(rot_new)), unit_circle_radius*math.cos(math.radians(rot_new)))

    # new_unit_circle_grasp = (unit_circle_radius*math.sin(math.radians(rot_new)), unit_circle_radius*math.cos(math.radians(rot_new)))
    print("new_unit_circle_grasp: ", new_unit_circle_grasp)
    # print("alternate: ", (unit_circle_radius*math.cos(math.radians(rot_new)), unit_circle_radius*math.sin(math.radians(rot_new))))
    # new_unit_circle_grasp = (new_unit_circle_grasp[0], 2*new_unit_circle_grasp[1])
    new_global_grasp = (center[0] + new_unit_circle_grasp[0], center[1] + new_unit_circle_grasp[1])
    print("new_global_grasp: ", new_global_grasp)

    # new_global_grasp = (center[0] - new_unit_circle_grasp[0], center[1] - new_unit_circle_grasp[1])
    # new_global_grasp = (center[0] - new_unit_circle_grasp[0], -center[1] - new_unit_circle_grasp[1])
    x = new_global_grasp[0]
    y = new_global_grasp[1]
    rz = action[3] + rot
    rz = wrap_rz(rz)
    action_aug = np.array([x, y, action[2], rz, action[4]])
    print("action aug: ", action_aug)

    return pcl_aug, action_aug, goal_aug

def visualize_pred_action_sequence(action, state, goal, ctr_path=None):
    """
    action: unnormalized action
    state: centered and scaled point cloud
    """
    # print("\n\naction: ", action)
    # action = unnormalize_a(action)  
    # print("unnormalized a: ", action)
    action = action.copy()

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