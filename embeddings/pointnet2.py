import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from embeddings.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

class PointNet2(nn.Module):
    def __init__(self, normal_channel=False):
        super(PointNet2, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)

    def forward(self, pcl):
        B, _, _ = pcl.shape
        print("pcl shape: ", pcl.shape)
        pcl = pcl.permute(0, 2, 1) # NEEDS TO BE SHAPE (Batch, N, 3)
        print("pcl shape: ", pcl.shape)
        if self.normal_channel:
            norm = pcl[:, 3:, :]
            pcl = pcl[:, :3, :]
        else:
            norm = None
        print("pcl shape: ", pcl.shape)
        print("norm shape: ", norm)
        l1_xyz, l1_points = self.sa1(pcl, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        return l3_points
        
    
class PointNetProjection(nn.Module):
    def __init__(self, input_dim=1024, latent_dim=512):
        super(PointNetProjection, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(1024, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, latent_dim)

    def forward(self, l3_points):
        B = l3_points.shape[0] # TODO: verify this is correct
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x,l3_points