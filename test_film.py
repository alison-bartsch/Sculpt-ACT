import torch
import json
import os
import numpy as np
from embeddings.embeddings import EncoderHead, EncoderHeadFiLM, EncoderHeadFiLMPretrained
from pointBERT.tools import builder
from pointBERT.utils.config import cfg_from_yaml_file

# load the pointbert encoder
device = torch.device('cuda')
config = cfg_from_yaml_file('pointBERT/cfgs/PointTransformer.yaml')
model_config = config.model
pointbert = builder.model_builder(model_config)
weights_path = 'pointBERT/point-BERT-weights/Point-BERT.pth'
pointbert.load_model_from_ckpt(weights_path)
pointbert.to(device)

# load the pretrained film encoder
enc_checkpoint = torch.load('pointBERT/encoder_weights/checkpoint', map_location=torch.device('cpu'))
projection_head = enc_checkpoint['encoder_head'].to(device)

# import two pointclouds
goal = np.load('X_target.npy')
goal = torch.from_numpy(goal).to(torch.float32)
goals = torch.unsqueeze(goal, 0).to(device)

state = np.load('X_target.npy')
state = torch.from_numpy(state).to(torch.float32)
states = torch.unsqueeze(state, 0).to(device)

# try passing them through the FiLM encoder after passing through pointbert
# encoded_dim = 768
# latent_dim = 512
# film_head = EncoderHeadFiLMPretrained(encoded_dim, latent_dim, projection_head, encoded_dim).to(device)
film_checkpoint = torch.load()
film_head = enc_checkpoint['encoder_head'].to(device)

embed_goal = pointbert(goals)
embed_state = pointbert(states)

film_state = film_head(embed_state, embed_goal)
enco_goal = projection_head(embed_goal)
print("film_state shape: ", film_state.shape)
print("enco_goal shape: ", enco_goal.shape)

# load the policy
ckpt_dir = 'checkpoints/X_c'
ckpt_name = ''
with open(ckpt_dir + '/policy_config.json') as json_file:
    policy_config = json.load(json_file)
ckpt_path = os.path.join(ckpt_dir, ckpt_name)
policy = make_policy('ACT', policy_config)
loading_status = policy.load_state_dict(torch.load(ckpt_path))
print(loading_status)
policy.cuda()
policy.eval()
print(f'Loaded: {ckpt_path}')