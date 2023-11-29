import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionNetwork(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(ActionNetwork, self).__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # two state inputs
        self.action_predictor = nn.Sequential(
            nn.Linear(self.latent_dim*2, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )

    def forward(self, ls, lns):
        x = torch.cat((ls, lns), dim=-1)
        # x = lns - ls
        pred_actions = self.action_predictor(x)
        return pred_actions

class EncoderHead(nn.Module):
    def __init__(self, encoded_dim, latent_dim):
        super(EncoderHead, self).__init__()
        self.encoded_dim = encoded_dim
        self.latent_dim = latent_dim

        self.encoder_head = nn.Sequential(
            nn.Linear(self.encoded_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim))

    def forward(self, encoded_pcl):
        x = torch.cat([encoded_pcl[:,0], encoded_pcl[:, 1:].max(1)[0]], dim = -1) # concatenation strategy from pointtransformer
        # x = torch.flatten(encoded_pcl, start_dim=1)
        latent_state = self.encoder_head(x)
        return latent_state