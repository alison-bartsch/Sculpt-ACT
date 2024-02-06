import math
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE

from pointBERT.tools import builder
from pointBERT.utils.config import cfg_from_yaml_file

vis = 'Diffusion' # ['pretrained', 'ACT', 'Diffusion', 'VINN]

data_path = '/home/alison/Clay_Data/Trajectory_Data/No_Aug_Dec14_Human_Demos/X'

if vis == 'pretrained':
    # load in pointbert encoder from weights pretrained on ShapeNet dataset
    device = torch.device('cuda')
    config = cfg_from_yaml_file('pointBERT/cfgs/PointTransformer.yaml')
    model_config = config.model
    pointbert = builder.model_builder(model_config)
    weights_path = 'pointBERT/point-BERT-weights/Point-BERT.pth'
    pointbert.load_model_from_ckpt(weights_path)
    pointbert.to(device)
elif vis == 'ACT':
    # ckpt_dir = '/home/alison/Documents/GitHub/Sculpt-ACT/checkpoints/exp4_new_dataloarder_more_epochs_delta_test'
    ckpt_dir = '/home/alison/Documents/GitHub/Sculpt-ACT/checkpoints/exp10_dataloadersanitycheck_pastaction_negpos'
    # load the pointbert and projection head models from ACT checkpoint
    device = torch.device('cuda')
    enc_checkpoint = torch.load(ckpt_dir + '/encoder_best_checkpoint', map_location=torch.device('cpu'))
    encoder_head = enc_checkpoint['encoder_head'].to(device)
    config = cfg_from_yaml_file('pointBERT/cfgs/PointTransformer.yaml')
    model_config = config.model
    pointbert = builder.model_builder(model_config)
    weights_path = ckpt_dir + '/best_pointbert.pth'
    pointbert.load_model_from_ckpt(weights_path)
    pointbert.to(device)
elif vis == 'Diffusion':
    ckpt_dir = '/home/alison/Documents/GitHub/diffusion_policy_3d/checkpoints/run_more_epochs'
    # load the pointbert and projection head models from ACT checkpoint
    device = torch.device('cuda')
    enc_checkpoint = torch.load(ckpt_dir + '/encoder_best_checkpoint', map_location=torch.device('cpu'))
    encoder_head = enc_checkpoint['encoder_head'].to(device)
    config = cfg_from_yaml_file('pointBERT/cfgs/PointTransformer.yaml')
    model_config = config.model
    pointbert = builder.model_builder(model_config)
    weights_path = ckpt_dir + '/best_pointbert.pth'
    pointbert.load_model_from_ckpt(weights_path)
    pointbert.to(device)

# dictionary with length of each trajectory (i.e. number of states -- number of actions = n_states - 1)
traj_dict = {0: 4,
             1: 7,
             2: 8,
             3: 8,
             4: 8,
             5: 8,
             6: 8,
             7: 9,
             8: 8,
             9: 8}

state_labels = []
traj_labels = []
rot_labels = []
embedded_pcls = []
rot_aug = 3
# iterate through trajectories
for i in tqdm(range(10)):
    # iterate through states in trajectory
    for s in range(traj_dict[i]):
        state = np.load(data_path + '/Trajectory' + str(i) + '/unnormalized_state' + str(s) + '.npy')
        center = np.load(data_path + '/Trajectory' + str(i) + '/pcl_center' + str(s) + '.npy')
        # iterate through rotation augmentations
        n_aug = int(360/rot_aug)
        for j in range(n_aug):
            idx = i*n_aug + j
            # apply rotation augmentation to the state point cloud and center and normalize
            rot = rot_aug * j
            rot_state = state - center
            R = o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz((0, 0, math.radians(rot)))
            rot_state = R @ rot_state.T
            pcl_aug = rot_state.T + center
            processed_state = (pcl_aug - center) * 10.0

            # embed the point cloud
            processed_state = torch.from_numpy(processed_state).to(torch.float32)
            states = torch.unsqueeze(processed_state, 0).to(device).contiguous()
            tokenized_states = pointbert(states)

            if vis == 'pretrained':
                pcl_embed = torch.cat([tokenized_states[:,0], tokenized_states[:, 1:].max(1)[0]], dim = -1)
            elif vis == 'ACT' or vis == 'Diffusion':
                pcl_embed = encoder_head(tokenized_states)

            # append to lists
            state_labels.append(s) # where s is the state index
            traj_labels.append(idx)
            rot_labels.append(rot)
            embedded_pcls.append(pcl_embed.cpu().detach().numpy())

data = np.squeeze(np.array(embedded_pcls))
print("Data shape: ", data.shape)
tsne_pcls = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(data)

# plot tsne with state index within trajectory
plt.scatter(tsne_pcls[:,0], tsne_pcls[:,1], c=state_labels)
plt.colorbar().set_label('State Index')
plt.show()

# plot tsne with trajectory indices
plt.scatter(tsne_pcls[:,0], tsne_pcls[:,1], c=traj_labels)
plt.colorbar().set_label('Trajectory Index')
plt.show()

# plot tsne with rotation augmentations
plt.scatter(tsne_pcls[:,0], tsne_pcls[:,1], c=rot_labels)
plt.colorbar().set_label('Rotation Augmentation')
plt.show()