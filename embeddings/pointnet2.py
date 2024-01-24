import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from embeddings.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

class PointNet2(nn.Module):
    def __init__(self, normal_channel=False):
        super(PointNet2, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        # swap last two dimensions of xyz
        xyz = xyz.permute(0, 2, 1) # NEEDS TO BE SHAPE (Batch, 3, N)
        # xyz = xyz[:,:,0:1024]
        # print("xyz shape: ", xyz.shape)
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        xyz = xyz[:,]
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        return l3_points

        # x = l3_points.view(B, 1024)
        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        # x = self.fc3(x)
        # x = F.log_softmax(x, -1)


        # return x, l3_points
# class PointNet2(nn.Module):
#     def __init__(self, normal_channel=False):
#         super(PointNet2, self).__init__()
#         in_channel = 3 if normal_channel else 0
#         self.normal_channel = normal_channel
#         self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
#         self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
#         self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)

#     def forward(self, pcl):
#         B, _, _ = pcl.shape
#         print("pcl shape: ", pcl.shape)
#         pcl = pcl.permute(0, 2, 1) # NEEDS TO BE SHAPE (Batch, N, 3)
#         print("pcl shape: ", pcl.shape)
#         if self.normal_channel:
#             norm = pcl[:, 3:, :]
#             pcl = pcl[:, :3, :]
#         else:
#             norm = None
#         print("pcl shape: ", pcl.shape)
#         print("norm shape: ", norm)
#         l1_xyz, l1_points = self.sa1(pcl, norm)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         return l3_points
        
    
class PointNetProjection(nn.Module):
    def __init__(self, input_dim=1024, latent_dim=512):
        super(PointNetProjection, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(1024, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, latent_dim)

    def forward(self, l3_points):
        B = l3_points.shape[0] # TODO: verify this is correct
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x # ,l3_points