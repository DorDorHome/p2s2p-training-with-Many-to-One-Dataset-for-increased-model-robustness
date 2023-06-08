#Alvin: an additional dataset subclass was created, for preprocessing purpose


from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils



class ImagesDataset(Dataset):
	"""
	The original ImagesDataset object. 
	To make a dataset from source_root (input) and target_root (intended output)
	
	"""

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root)) # this simply create a list of the paths to each image
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		return from_im, to_im



class ManyToOneDataset(Dataset):
	# added by Alvin, for the purpose of building many to one datasets
	""""

	To make a dataset from source_root (input) and target_root (intended output)



	"""
	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
		print('\n initiating a Many to One dataset.\n')
		self.source_paths = sorted(data_utils.make_dataset(source_root)) # this simply create a list of the paths to each image
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

		if 'fixed_ration' in vars(self.opts):
			assert 'ratio' in vars(self.opts), "Please include 'ratio' in the opts"
	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		"""
		index is done on the source paths. 
		The corresponding index in target path is done by floor division
		Cross-checked


		"""

		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

		# one, unreliable way is to simply 
		#to_path = self.target_paths[index//self.opts['ratio']]

		# a more reliable way is to strip the from_path of the last two elements:
		to_path = from_path[:-6] + '.jpg'
		to_path = to_path.replace('artiticial_shroud_dataset', 'CelebA-HQ-img' )
		# to check that the name of the file matches
		# print('from path:', from_path)
		# print('to_path: ', to_path)
		to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		return from_im, to_im




class SingleImagesDataset(Dataset):
	# added by Alvin

	def __init__(self, root, opts, transform=None):
		self.paths = sorted(data_utils.make_dataset(root)) # this simply create a list of the paths to each image
		self.transform = transform
		self.opts = opts

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		path = self.paths[index]
		im = Image.open(path)
		im = im.convert('RGB') if self.opts.label_nc == 0 else im.convert('L')

		if self.transform:
			im = self.transform(im)

		return im

class SingleImagesDataset_with_path(Dataset):
	"""
	This is an adaption to SingleImagesDataset so each item also get the path
	
	"""

	def __init__(self, root, opts, transform=None):
		self.paths = sorted(data_utils.make_dataset(root)) # this simply create a list of the paths to each image
		self.transform = transform
		self.opts = opts

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		path = self.paths[index]
		im = Image.open(path)
		im = im.convert('RGB') if self.opts.label_nc == 0 else im.convert('L')

		if self.transform:
			im = self.transform(im)

		return im, path