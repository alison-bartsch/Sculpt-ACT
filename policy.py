import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

        # build the Point-BERT model

    # def __call__(self, qpos, image, actions=None, is_pad=None):
    #     env_state = None
    #     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                      std=[0.229, 0.224, 0.225])
    #     image = normalize(image)
    #     if actions is not None: # training time
    #         actions = actions[:, :self.model.num_queries]
    #         is_pad = is_pad[:, :self.model.num_queries]

    #         print("\nModel num queries: ", self.model.num_queries)

    #         a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
    #         total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
    #         loss_dict = dict()
    #         all_l1 = F.l1_loss(actions, a_hat, reduction='none')
    #         l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
    #         loss_dict['l1'] = l1
    #         loss_dict['kl'] = total_kld[0]
    #         loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
    #         return loss_dict
    #     else: # inference time
    #         a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
    #         return a_hat
        
    def _cosine_similarity(self, x, y):
        """
        Cosine similarity as loss to encourage dissimilarity. 
        1 means identical, 0 means orthogonal. Calculating the 
        similarity for tensors of shape (1, actions)
        """
        return F.cosine_similarity(x, y, dim=0)
    
    def _dissimilar_loss(self, actions):
        """
        Calculating the dissimilarity loss for a batch of actions.
        This is to discourage the model from repeating action predictions.
        """
        loss = 0
        for i in range(actions.shape[0]):
            # iterate through the sequence prediction
            for j in range(actions.shape[1] - 1):
                loss += self._cosine_similarity(actions[i][j], actions[i][j+1])
        return loss

        # loss = 0
        # for i in range(actions.shape[0]):
        #     for j in range(i+1, actions.shape[0]):
        #         loss += self._cosine_dissimilarity(actions[i], actions[j])
        # return loss

    def __call__(self, goal, state, actions=None, is_pad=None, concat_goal=False, delta_goal=False, no_pos_embed=False):
        env_state = None

        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            a_hat, is_pad_hat, (mu, logvar) = self.model(goal, state, env_state, actions, is_pad, concat_goal, delta_goal, no_pos_embed)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            # modification to discourage repeate actions
            dissimilarity_weight = 0.1
            dissimilarity_loss = self._dissimilar_loss(actions)
            # print("dis loss: ", dissimilarity_loss)
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight + dissimilarity_weight * dissimilarity_loss
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(goal, state, env_state) # no action, sample from prior
            return a_hat

    def __call_prev__(self, qpos, state, actions=None, is_pad=None):
        env_state = None

        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            # print("\nModel num queries: ", self.model.num_queries)

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, state, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            # print("\nactions: ", actions.shape)
            # print("a_hat: ", a_hat.shape)
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, state, env_state) # no action, sample from prior
            return a_hat
        

    # def __call__(self, state, actions=None, is_pad=None):
    #     env_state = None
        
    #     if actions is not None: # training time
    #         actions = actions[:, :self.model.num_queries]
    #         is_pad = is_pad[:, :self.model.num_queries]

    #         print("\nModel num queries: ", self.model.num_queries)

    #         a_hat, is_pad_hat, (mu, logvar) = self.model(state, env_state, actions, is_pad)
    #         total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
    #         loss_dict = dict()
    #         all_l1 = F.l1_loss(actions, a_hat, reduction='none')
    #         l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
    #         loss_dict['l1'] = l1
    #         loss_dict['kl'] = total_kld[0]
    #         loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
    #         return loss_dict
    #     else: # inference time
    #         a_hat, _, (_, _) = self.model(state, env_state) # no action, sample from prior
    #         return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
