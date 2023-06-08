from argparse import ArgumentParser

import os
import sys

sys.path.append(".")
sys.path.append("..")


# for setting up data type:
from configs import data_configs
from datasets.images_dataset import ImagesDataset, SingleImagesDataset
from options.train_options import TrainOptions, PreprocessOptions
from utils.common import tensor2im
from torchvision import utils



# create the parser
# parser = ArgumentParser(description='split the images from input folder into two and save in output folder')

# # add the arguments:
# parser.add_argument('input_folder',
#                     type = str,
#                     help = 'The folder that contains the images to be split'
#                     )


# parser.add_argument('right_half_result',
#                    type = str,
#                     help = 'The folder to contain the right hand side of the split images'
#                     ) 
                

# parser.add_argument('left_half_result',
#                    type = str,
#                     help = 'The folder to contain the left hand side of the split images'
#                     ) 

# # execute the parse_args() method
# # get a Namespace object that contains the user input.
# args = parser.parse_args()




# temp: for experimenting with argparser only:
# print(args.input_folder)
# print(args.right_half_result)

# setting up opts:

opts = PreprocessOptions().parse()




# load the data from input_folder:


# get the appropriate dictionary from the data_configs.py:
dataset_args = data_configs.DATASETS[opts.dataset_type]

print(dataset_args.items())

print(dataset_args['transforms'](opts = opts))


transforms_dict = dataset_args['transforms'](opts = opts).get_transforms()

train_dataset = SingleImagesDataset(root=dataset_args['to_preprocess_train'],
                                transform= transforms_dict['transform_to_preprocess'],
                                opts=opts)

print('size of the dataset: ', len(train_dataset))


print(type(train_dataset[0])) # .shape)

print(train_dataset[0].shape)

image_wide = train_dataset[0].shape[2]

# show the source image:

tensor2im(train_dataset[0]).show()

# split the images:

image_testL, image_testR = train_dataset[0][:,:,:(image_wide//2)], train_dataset[0][:,:, (image_wide//2):]
#tensor2im(image_testL).show()
#tensor2im(image_testR).show()

# setup output path:

source_out_folder = opts.right_half_result
target_out_folder = opts.left_half_result

print('source folder: ',source_out_folder)
print('target folder: ', target_out_folder)

for i, image in enumerate(train_dataset):
    
    imageL, imageR = image[:,:,:512], image[:,:,512:]
    if i < 2:
        tensor2im(imageL).show()
    source_out_path = os.path.join(source_out_folder, f'{i}.png' )
    target_out_path = os.path.join(target_out_folder, f'{i}.png' )
    utils.save_image(
        imageR,
        source_out_path,
        nrow = 1, 
        normalize = True,
        range = (-1,1),
    )
    utils.save_image(
        imageL,
        target_out_path,
        nrow = 1, 
        normalize = True,
        range = (-1,1),
    )



# do the same thing for test set:


# test_dataset = SingleImagesDataset(root=dataset_args['to_preprocess_test'],
#                                 transform= transforms_dict['transform_to_preprocess'],
#                                 opts=opts)

# print('size of the dataset: ', len(test_dataset))


# print('size of the dataset: ', len(test_dataset))


# print(type(test_dataset[0])) # .shape)

# print(test_dataset[0].shape)

# # setup output path:

# source_out_folder = opts.right_half_result
# target_out_folder = opts.left_half_result

# for i, image in enumerate(test_dataset):
    
#     imageL, imageR = image[:,:,:512], image[:,:,512:]
#     if i < 2:
#         tensor2im(imageL).show()
#     source_out_path = os.path.join(source_out_folder, f'{i}.png' )
#     target_out_path = os.path.join(target_out_folder, f'{i}.png' )
#     utils.save_image(
#         imageR,
#         source_out_path,
#         nrow = 1, 
#         normalize = True,
#         range = (-1,1),
#     )
#     utils.save_image(
#         imageL,
#         target_out_path,
#         nrow = 1, 
#         normalize = True,
#         range = (-1,1),
#         )







# testing_path = sys.argv



# this part is not needed. For learning purpose

# if len(sys.argv) > 2:
#     print('You have specified too many arguments')
#     sys.exit()

# if len(sys.argv) < 2:
#     print('You need to specify the path to be listed')
#     sys.exit()

# print(testing_path[1])