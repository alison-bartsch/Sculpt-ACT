import numpy as np
import open3d as o3d
import robomail.vision as vis

# pointcloud = np.load('/home/alison/Clay_Data/Trajectory_Data/Aug24_Human_Demos/X/Trajectory0/state4.npy')

# pcl = o3d.geometry.PointCloud()
# pcl.points = o3d.utility.Vector3dVector(pointcloud)
# pcl_colors = np.tile(np.array([0,0,1]), (len(pointcloud),1))
# pcl.colors = o3d.utility.Vector3dVector(pcl_colors)
# o3d.visualization.draw_geometries([pcl])

# goal plot
goal_pcl = np.load('X_target.npy')

pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(goal_pcl)
pcl_colors = np.tile(np.array([1,0,0]), (len(goal_pcl),1))
pcl.colors = o3d.utility.Vector3dVector(pcl_colors)

# initialize the cameras
cam2 = vis.CameraClass(2)
cam3 = vis.CameraClass(3)
cam4 = vis.CameraClass(4)
cam5 = vis.CameraClass(5)

# initialize the 3D vision code
pcl_vis = vis.Vision3D()

_, _, pc2, _ = cam2._get_next_frame()
_, _, pc3, _ = cam3._get_next_frame()
_, _, pc4, _ = cam4._get_next_frame()
_, _, pc5, _ = cam5._get_next_frame()
pointcloud = pcl_vis.fuse_point_clouds(pc2, pc3, pc4, pc5, vis=False)

pcl_o3d = o3d.geometry.PointCloud()
pcl_o3d.points = o3d.utility.Vector3dVector(pointcloud)
pcl_o3d_colors = np.tile(np.array([0,0,1]), (len(pointcloud),1))
pcl_o3d.colors = o3d.utility.Vector3dVector(pcl_o3d_colors)

# hard coded center
pc_center = 10*np.array([0.6, 0.0, 0.24])
pc_center = np.expand_dims(pc_center, axis=0)
ctr = o3d.geometry.PointCloud()
ctr.points = o3d.utility.Vector3dVector(pc_center)
ctr_colors = np.tile(np.array([0,1,0]), (len(pc_center),1))
ctr.colors = o3d.utility.Vector3dVector(ctr_colors)
o3d.visualization.draw_geometries([pcl, pcl_o3d, ctr])

# get the centers of each point cloud
goal_np = np.asarray(pcl.points)
state_np = np.asarray(pcl_o3d.points)
print("\nMean Goal: ", np.mean(goal_np, axis=0))
print("Mean State: ", np.mean(state_np, axis=0))

# get the mins of each point cloud
print("\nMin Goal: ", np.min(goal_np, axis=0))
print("Min State: ", np.min(state_np, axis=0))

# get the maxs of each point cloud
print("\nMax Goal: ", np.max(goal_np, axis=0))
print("Max State: ", np.max(state_np, axis=0))

# reprocess the clouds centering about zeros
goal_ctr = pcl.get_center()
state_ctr = pcl_o3d.get_center()
print("\nGoal Center: ", goal_ctr)
print("State Center: ", state_ctr)

pcl = o3d.geometry.PointCloud()
goal = goal_pcl - goal_ctr
pcl.points = o3d.utility.Vector3dVector(goal)
pcl_colors = np.tile(np.array([1,0,0]), (len(goal),1))
pcl.colors = o3d.utility.Vector3dVector(pcl_colors)

pcl_o3d = o3d.geometry.PointCloud()
state = pointcloud - state_ctr
pcl_o3d.points = o3d.utility.Vector3dVector(state)
pcl_o3d_colors = np.tile(np.array([0,0,1]), (len(state),1))
pcl_o3d.colors = o3d.utility.Vector3dVector(pcl_o3d_colors)

o3d.visualization.draw_geometries([pcl, pcl_o3d])