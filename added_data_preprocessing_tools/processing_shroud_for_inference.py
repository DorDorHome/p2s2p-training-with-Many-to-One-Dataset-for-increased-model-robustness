from argparse import ArgumentParser

import os
from random import gauss
import sys
from xml.etree.ElementTree import tostring
sys.path.append(".")
sys.path.append("..")

from configs.transforms_config import TransformOriginalShroud



# for setting up data type:
from configs import data_configs
from datasets.images_dataset import ImagesDataset, SingleImagesDataset
from options.train_options import PreprocessJesusOptions
from utils.common import tensor2im
from torchvision import utils



# setting up opts:

opts = PreprocessJesusOptions().parse()

print('\opts is ', opts, '\n\n')
print('opts.dataset_type', opts.dataset_type, '\n') 
# get the appropriate dictionary from the data_configs.py:
dataset_args = data_configs.DATASETS[opts.dataset_type]

print(dataset_args.items())

print(dataset_args['transforms'](opts = opts))

# several transform are available with transform_dict. See the file transforms_config.TransformOriginalShroud
transforms_dict = dataset_args['transforms'](opts = opts).get_transforms()

print('\ntransform_dict is', transforms_dict)

# load the dataset:
original_jesus_dataset = SingleImagesDataset(root=opts.original_shroud_input_folder,
                        transform=transforms_dict['transform_initial'],
                        opts=opts )

blur_jesus_dataset = SingleImagesDataset(root=opts.original_shroud_input_folder,
                        transform=transforms_dict['transform_with_blur'],
                        opts=opts )

print('size of the original dataset: ', len(original_jesus_dataset))

print(type[original_jesus_dataset[0]])

print(original_jesus_dataset[0].shape)

# show the source image:

#tensor2im(original_jesus_dataset[0]).show()

# show the blur_jesus image (first one):

#tensor2im(blur_jesus_dataset[0]).show()


# save all the blur jesus image in the intended output folder:

for i, image in enumerate(blur_jesus_dataset):
    #tensor2im(image).show()
    gauss_kernel_size = str(opts.Gaussian_blur_kernel_size)
    gauss_std = str(opts.Gaussian_blur_sigma)
    denoise_amount= str(opts.denoise_amount)
    bw = str(opts.convert_to_BW)
    image_path = os.path.join(opts.transformed_shroud_output_folder, f'jesus_{i}_gauss_ker_{gauss_kernel_size}_std_{gauss_std}_denoise_{denoise_amount}_BW_{bw}.png')


    utils.save_image(
        image,
        image_path,
        nrow = 1, 
        normalize = True,
        range = (-1,1)
     )