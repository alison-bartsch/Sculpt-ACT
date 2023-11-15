import numpy as np
import open3d as o3d

pointcloud = np.load('/home/alison/Clay_Data/Trajectory_Data/Aug24_Human_Demos/X/Trajectory0/state4.npy')

pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(pointcloud)
pcl_colors = np.tile(np.array([0,0,1]), (len(pointcloud),1))
pcl.colors = o3d.utility.Vector3dVector(pcl_colors)
o3d.visualization.draw_geometries([pcl])