import torch
import numpy as np
import open3d as o3d
from imitate_clay_episodes import load_clay_data
# from imitate_clay_episodes import *
from action_geometry_utils import *

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

    o3d.visualization.draw_geometries([og_pcl, g_pcl, line_set, g1, g2]) 

dataset_dir = '/home/alison/Clay_Data/Trajectory_Data/Aug24_Human_Demos/X'
num_episodes = 899
batch_size_train = 1
batch_size_val = 1
action_pred = True

train_dataloader, val_dataloader = load_clay_data(dataset_dir, num_episodes, batch_size_train, batch_size_val, action_pred, visualize_grasp)
for batch_idx, data in enumerate(train_dataloader):
    goal_data, state_data, action_data, is_pad = data
    