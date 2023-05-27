# # this file was created by Alvin Chan
# # to explore the psp and stylegan2 model

# import os
# import argparse
# from argparse import Namespace

# from tqdm import tqdm
# import time
# import numpy as np
# import torch
# from PIL import Image
# from torch.utils.data import DataLoader
# import sys



# sys.path.append(".")
# sys.path.append("..")


# # from models.stylegan2.model import Generator
# from models.psp import pSp

# import matplotlib
# matplotlib.use('Agg')
# import math


import os
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

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp
from torchvision import utils



def run():
    test_opts = TestOptions().parse()

    # create folders to save generated images:

    print('-- exp_dir is', test_opts.exp_dir)
    if test_opts.resize_factors is not None:
        # this part is usually not required. Inherited from the super-resolution inference task only.
        assert len(
            test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
        
        decoder_results = os.path.join(test_opts.exp_dir, 'psp_decoder(stylegan2)',
                                    'downsampling_{}'.format(test_opts.resize_factors))
        encoder_results = os.path.join(test_opts.exp_dir, 'psp_encoder',
                                        'downsampling_{}'.format(test_opts.resize_factors))
    else:
        decoder_results = os.path.join(test_opts.exp_dir, 'psp_decoder(stylegan2)',
                                        'output')
        encoder_results = os.path.join(test_opts.exp_dir, 'psp_encoder', 'output')

    os.makedirs(decoder_results, exist_ok=True)
    os.makedirs(encoder_results, exist_ok=True)


    device = 'cuda'
    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location= device) #'cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)
    print('checkpoint dictionary keys: ', ckpt.keys())


    net = pSp(opts)
    net.eval()
    net.cuda()


    # define the stylegan2 to be the decoder part of the psp net:
    g_ema = net.decoder



    
    # setting for generating images:
    # it should be added to the test_options.py file
    n_pics = 4
    truncation = 1
    n_sample = 1 # number of samples to be generated for each image
    truncation_mean = 4096 # number of randomly generated w to form the 'average' face


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
            sample_z = torch.randn(n_sample, 512 , device=device)
            # sample_z = torch.randn(args.n_sample, args.latent, device=device)
            sample, _ = g_ema(
                [sample_z])
                
            #     , truncation= truncation, truncation_latent=mean_latent
            # )

            # print(sample)
            
            output_samples_path = os.path.join(decoder_results, f'{str(i).zfill(6)}.png')


            utils.save_image(
                sample,
                output_samples_path,
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

if __name__ == '__main__':
    run()
