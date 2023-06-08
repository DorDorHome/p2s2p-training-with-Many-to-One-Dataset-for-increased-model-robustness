
import os
import json
import sys
import pprint
sys.path.append(".")
sys.path.append("..")
# from torch.utils.data import Dataset
# from PIL import Image
# from utils import data_utils
from datasets.images_dataset import ManyToOneDataset


from options.train_options import TrainOptions
#from training.coach import Coach


def main():
    opts = TrainOptions().parse()
    source_root = '/Users/shufaichan/Documents/datasets/CelebAMask-HQ/CelebA-HQ-img'
    target_root = '/Users/shufaichan/Documents/datasets/CelebAMask-HQ/artiticial_shroud_dataset'

    # opts['fixed_ratio']= True
    # opts['ratio'] =4



    dataset = ManyToOneDataset(source_root, target_root, opts, target_transform=None, source_transform=None)

    # checking the paths agrees:

    print(dataset[1])

    # displaying the corresponding images:

if __name__=='__main__':
    main()