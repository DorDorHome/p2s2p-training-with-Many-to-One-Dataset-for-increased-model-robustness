# this file was created by Alvin Chan
# to explore the stylegan2 model
import os
import argparse
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys



sys.path.append(".")
sys.path.append("..")


from models.stylegan2.model import Generator

import matplotlib
matplotlib.use('Agg')
import math


from torch import nn
from models.encoders import psp_encoders
from models.stylegan2.model import Generator
from configs.paths_config import model_paths
from torchvision import utils

# def generate(args, g_ema, device, mean_latent):

#     with torch.no_grad():
#         g_ema.eval()
#         for i in tqdm(range(args.pics)):
#             sample_z = torch.randn(args.sample, args.latent, device=device)

#             sample, _ = g_ema(
#                 [sample_z], truncation=args.truncation, truncation_latent=mean_latent
#             )

#             utils.save_image(
#                 sample,
#                 f"sample/{str(i).zfill(6)}.png",
#                 nrow=1,
#                 normalize=True,
#                 range=(-1, 1),
#             )



# for setting up the model:
# pretrained_model = 'pretrained_models/550000.pt'
pretrained_model = "stylegan2-ffhq-config-f.pt"

pretrained_model_folder_path = 'pretrained_models'

pretrained_model_path = os.path.join(pretrained_model_folder_path, pretrained_model)

print('\n loading model checkpoint from:', pretrained_model_path, '\n')

device = 'cuda'
size = 1024
latent = 512
n_mlp =8
channel_mulipler = 2


# setting for generating images:
n_pics = 4
truncation = 1
n_sample = 1 # number of samples to be generated for each image
truncation_mean = 4096 # number of randomly generated w to form the 'average' face



g_ema = Generator(
    size= size, style_dim = latent, n_mlp = n_mlp , channel_multiplier= channel_mulipler
).to(device)

checkpoint=  torch.load(pretrained_model_path)

print('checkpoint dictionary keys: ', checkpoint.keys())

g_ema.load_state_dict(checkpoint['g_ema'], strict= False)

# create folders to save generated images:

out_path = "path_to_experiment/vanilla_stylegan2/generated_sample"

os.makedirs(out_path, exist_ok= True)

# forward pass:

if truncation < 1: 
    with torch.no_grad():
        mean_latent = g_ema.mean_latent(truncation_mean)
else:
    mean_latent = None


# forward pass:

with torch.no_grad():
    g_ema.eval()
    for i in tqdm(range(n_pics)):
        sample_z = torch.randn(n_sample, latent, device=device)
        # sample_z = torch.randn(args.n_sample, args.latent, device=device)
        sample, _ = g_ema(
            [sample_z])
            
        #     , truncation= truncation, truncation_latent=mean_latent
        # )

        # print(sample)

        utils.save_image(
            sample,
            f"path_to_experiment/vanilla_stylegan2/generated_sample/{str(i).zfill(6)}_with_{pretrained_model}.png",
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )