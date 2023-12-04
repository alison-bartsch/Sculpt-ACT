import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMBlock(nn.Module):
    def __init__(self):
        super(FiLMBlock, self).__init__()

    def forward(self, x, gamma, beta):
        return gamma * x + beta


class EncoderHeadFiLM(nn.Module):
    def __init__(self, encoded_dim, latent_dim, condition_dim):
        """
        New Projection Head for Point-BERT with a built-in FiLM block for
        goal conditioning (affine transformation conditioning).
        """
        super(EncoderHeadFiLM, self).__init__()
        self.encoded_dim = encoded_dim
        self.latent_dim = latent_dim

        # NOTE: may need to have a separae film generator for each layer because the dimensions vary!!!
        # self.film_generator = [nn.Linear(condition_dim, 2 * self.encoded_dim).cuda(),
        #                        nn.Linear(condition_dim, 2 * self.latent_dim).cuda()]
        self.film_generator = nn.Linear(condition_dim, 2 * self.latent_dim).cuda()

        self.fc1 = nn.Linear(self.encoded_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, self.latent_dim)
        
        self.gelu = nn.GELU()
        self.film = FiLMBlock()

        # self.encoder_head = nn.Sequential(
        #     nn.Linear(self.encoded_dim, self.latent_dim),
        #     nn.GELU(),
        #     nn.Linear(self.latent_dim, self.latent_dim))

    def forward(self, encoded_pcl, encoded_goal):
        x = torch.cat([encoded_pcl[:,0], encoded_pcl[:, 1:].max(1)[0]], dim = -1) # concatenation strategy from pointtransformer
        goal = torch.cat([encoded_goal[:,0], encoded_goal[:, 1:].max(1)[0]], dim = -1)

        # film_param = self.film_generator[0](goal)
        film_param = self.film_generator(goal)
        gamma = film_param[:, :self.latent_dim]
        beta = film_param[:, self.latent_dim:]
        x = self.fc1(x)
        x = self.film(x, gamma, beta)
        x = self.gelu(x)

        # film_param = self.film_generator[1](goal)
        film_param = self.film_generator(goal)
        gamma = film_param[:, :self.latent_dim]
        beta = film_param[:, self.latent_dim:]
        x = self.fc2(x)
        x = self.film(x, gamma, beta)
        x = self.gelu(x)

        x = self.fc3(x)
        return x
    
class EncoderHeadFiLMPretrained(nn.Module):
    def __init__(self, encoded_dim, latent_dim, encoder, condition_dim):
        super(EncoderHeadFiLMPretrained, self).__init__()
        self.encoded_dim = encoded_dim
        self.latent_dim = latent_dim
        self.encoder = encoder.encoder_head
        self.condition_dim = condition_dim

        self.film_generator = nn.Linear(condition_dim, 2 * self.latent_dim).cuda()
        self.film = FiLMBlock()

    def forward(self, encoded_pcl, encoded_goal):
        x = torch.cat([encoded_pcl[:,0], encoded_pcl[:, 1:].max(1)[0]], dim = -1) # concatenation strategy from pointtransformer
        goal = torch.cat([encoded_goal[:,0], encoded_goal[:, 1:].max(1)[0]], dim = -1)

        film_param = self.film_generator(goal)
        gamma = film_param[:, :self.latent_dim]
        beta = film_param[:, self.latent_dim:]
        x = self.encoder[0](x) # linear layer
        x = self.film(x, gamma, beta)
        x = self.encoder[1](x) # gelu layer
        x = self.encoder[2](x) # final linear layer
        return x

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