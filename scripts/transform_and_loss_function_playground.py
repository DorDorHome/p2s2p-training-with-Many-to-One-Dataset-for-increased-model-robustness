# created by Alvin Chan
# to play with different loss functions 
# and transform (for blurring outputs)

# first, studying the properties of transform

# question: how to apply downsampling in the tensor, and have the same effect as
# the transform in the PIL image?

from argparse import Namespace
import os
import json
import sys
import pprint
from tracemalloc import DomainFilter
import torch

sys.path.append(".")
sys.path.append("..")

from datasets.inference_dataset import InferenceDataset

from options.train_options import TrainOptions
# from training.coach import Coach


from utils import common, train_utils
from criteria import id_loss, w_norm, moco_loss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
from models.psp import pSp
from training.ranger import Ranger
from options.test_options import TestOptions
from utils.common import tensor2im, log_input_image
from torchvision import utils

# for defining the transform class directly in this file
# used in step 3:
import torchvision.transforms as transforms
from datasets import augmentations
from datasets import augmentations_in_tensor


# plan:
# step 1: load data with blurry transform
# step 2: load data with original

# step 3: create a blurry transformation in the tensor Domain
# step 4: transform the original image in the tensor domain
# step 5: check, by using l2 norm, that the two results are the same:





#setting the opts:

test_opts = TestOptions().parse()

print(test_opts)

if test_opts.resize_factors is not None:
    assert len(
        test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
    out_path_results = os.path.join(test_opts.exp_dir, 'transform_playground_output',
                                    'downsampling_{}'.format(test_opts.resize_factors))
    out_path_coupled = os.path.join(test_opts.exp_dir, 'transform_playground_coupled',
                                    'downsampling_{}'.format(test_opts.resize_factors))
else:
    out_path_results = os.path.join(test_opts.exp_dir, 'transform_playground_output')
    out_path_coupled = os.path.join(test_opts.exp_dir, 'transform_playground_coupled')

os.makedirs(out_path_results, exist_ok=True)
os.makedirs(out_path_coupled, exist_ok=True)

# update test options with options used during training
ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
opts = ckpt['opts']
opts.update(vars(test_opts))
if 'learn_in_w' not in opts:
    opts['learn_in_w'] = False
if 'output_size' not in opts:
    opts['output_size'] = 1024
opts = Namespace(**opts)

print('\n opts: \n  ', opts)

# step 1: load data with blurry transform:

print(opts.dataset_type) # expect 'celebs_super_resolution'


if opts.dataset_type not in data_configs.DATASETS.keys():
    Exception(f'{opts.dataset_type} is not a valid dataset_type')
print(f'Loading dataset for {opts.dataset_type}')

# get the appropriate dictionary from the data_configs.py:
dataset_args = data_configs.DATASETS[opts.dataset_type]
print('dataset_args, which consists of paths to the datasets', dataset_args.items())

# first retrieve the transformation in for the required task (e.g. celebs_super_resolution)
# from the dictionary in the data_configs.py
# then, using the appropriate class (e.g. SuperResTransforms) defined in the transforms_config.py
# get the transforms_dict:
print('downsized factors (used in the transform_source ):', opts.resize_factors )
transforms_dict = dataset_args['transforms'](opts).get_transforms()

# added by Alvin for debugging purpose:
print("debug: path of source_root", dataset_args['train_source_root'])

# load the data. For details on the transform, see configs\transforms_configs.py: 
# The result is a subclass of Dataset, where __getitm__ gives a pair ( source_img, target_img):
# referring to InferenceDataset Class in inference_dataset.py file, each sample of this class
blurred_dataset = InferenceDataset(root=opts.data_path,
                            transform=transforms_dict['transform_inference'],
                            opts=opts)




# load in dataloader. Not required for now:
# blurred_dataloader = DataLoader(blurred_dataset,
#                             batch_size=opts.test_batch_size,
#                             shuffle=False,
#                             num_workers=int(opts.test_workers),
#                             drop_last=True)


# step 2: load original data, resized and normalized, ToTensor:

original_dataset = InferenceDataset(root = opts.data_path,
                                    transform = transforms_dict['transform_test'],
                                    opts = opts)

# step 2b; checking both images:
i = 0
output_original_samples_path = os.path.join(out_path_results, f'original_image_{i}.png')

print('type of original dataset member: ', type(original_dataset[i]))

utils.save_image(
    original_dataset[i],
    output_original_samples_path,
    nrow = 1, 
    normalize = True,
    range = (-1,1),
)

output_blurred_samples_path = os.path.join(out_path_results, f'image_blurred_when_loaded{i}.png')

utils.save_image(
    blurred_dataset[i],
    output_blurred_samples_path,
    nrow = 1, 
    normalize = True,
    range = (-1,1),
)


# step 3: create a blurry transformation in the tensor Domain
# help: the transformed used in the image loading is:
# 			'transform_test': transforms.Compose([
# 				transforms.Resize((256, 256)),
# 				transforms.ToTensor(),
# 				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),

# and: 
# 			'transform_source': transforms.Compose([
# 				transforms.Resize((256, 256)),
# 				augmentations.BilinearResize(factors=factors),
# 				transforms.Resize((256, 256)),
# 				transforms.ToTensor(),
# 				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
if opts.resize_factors is None:
    opts.resize_factors = '1,2,4,8,16,32'

factors = [int(f) for f in opts.resize_factors.split(",")]
print('step 3: factors: ', factors)
blur_transform_in_tensor = transforms.Compose([
                                transforms.Normalize([0,0,0], [2,2,2]),
                                transforms.Normalize([-0.5, -0.5, 0.5], [1,1,1]),
                                transforms.ToPILImage(),
                                transforms.Resize((256, 256)),
                                augmentations.BilinearResize(factors=factors),
                                transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                ])

blur_directly_in_tensor = transforms.Compose([
                                transforms.Normalize([0,0,0], [2,2,2]),
                                transforms.Normalize([-0.5, -0.5, 0.5], [1,1,1]),
                                #transforms.Resize((256, 256)),
                                augmentations_in_tensor.BilinearResize_in_tensor(factors=factors),
                                #transforms.Resize((256, 256)),
                                # transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                ])

# step 4: transform the original image in the tensor domain
# note that the transform expects an input as tensor,
# it cannot take Dataset Class as input directly.

# for debug: 
print('shape of input: ', original_dataset[i].shape)
print('type after changing to numpy:', original_dataset[i].numpy().shape)

original_dataset_transformed_in_tensor = blur_transform_in_tensor(original_dataset[i])

original_dataset_transformed_directly_in_tensor = blur_directly_in_tensor(original_dataset[i])

# step 4b: saving the new tensor as image:

transformed_in_tensor_image_path = os.path.join(out_path_results, f'blurred_as_tensor_{i}.png')

utils.save_image(
original_dataset_transformed_in_tensor,
    transformed_in_tensor_image_path,
    nrow = 1, 
    normalize = True,
    range = (-1,1),
)

utils.save_image(
original_dataset_transformed_directly_in_tensor,
    transformed_in_tensor_image_path,
    nrow = 1, 
    normalize = True,
    range = (-1,1),
)

# step 5, comparison of the two tensor using mse loss:

difference = original_dataset_transformed_in_tensor - blurred_dataset[i]

print(difference)
print(torch.norm(difference, p = 'fro'))
print(torch.norm(original_dataset_transformed_in_tensor,p = 'fro'))
tensor2im(difference).show()

print('directly in tensor:')
difference2 = original_dataset_transformed_directly_in_tensor - blurred_dataset[i]
print(difference2)
print(torch.norm(difference2, p = 'fro'))
print(torch.norm(original_dataset_transformed_directly_in_tensor,p = 'fro'))
tensor2im(difference2).show()