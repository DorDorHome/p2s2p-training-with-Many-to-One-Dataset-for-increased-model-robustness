
from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import testing_dataset_object
from images_dataset import ManyToOneDataset

source_root = '/Users/shufaichan/Documents/datasets/CelebAMask-HQ/CelebA-HQ-img'
target_root = '/Users/shufaichan/Documents/datasets/CelebAMask-HQ/artiticial_shroud_dataset'
opts = {}
opts['fixed_ratio']= True
opts['ratio'] =4
opts.label_nc = 0


dataset = ManyToOneDataset(source_root, target_root, opts, target_transform=None, source_transform=None)

# checking the paths agrees:

dataset[1]

# displaying the corresponding images:
