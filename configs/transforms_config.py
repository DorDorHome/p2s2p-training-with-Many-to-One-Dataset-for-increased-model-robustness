# added one transform class, ShroudToFaceTransform for the purpose of doing shroud translation.
# this is adapted from SketchToImage so is next to it. Edit when necessary.

# note that one import was hashed. Might need to add it back

from abc import abstractmethod
# from msilib.schema import Shortcut
import torchvision.transforms as transforms
import torchvision.transforms as transforms


from datasets import augmentations


class TransformsConfig(object):

	def __init__(self, opts):
		self.opts = opts

	@abstractmethod
	def get_transforms(self):
		pass

# this is used in converting the original jesus shroud so that the input align more closely to the training set:
class TransformOriginalShroud(TransformsConfig):
	def __init__(self, opts):
		super(TransformOriginalShroud, self).__init__(opts) # just save the opts as self.opts.

	def get_transforms(self):
		transforms_dict = {
			'transform_initial':transforms.Compose([transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_with_blur':transforms.Compose([transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.GaussianBlur(kernel_size = self.opts.Gaussian_blur_kernel_size, sigma=self.opts.Gaussian_blur_sigma),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
			])

		}
		return transforms_dict


class EncodeTransforms(TransformsConfig):

	def __init__(self, opts):
		super(EncodeTransforms, self).__init__(opts)

	def get_transforms(self):
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source': None,
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		}
		return transforms_dict


class FrontalizationTransforms(TransformsConfig):

	def __init__(self, opts):
		super(FrontalizationTransforms, self).__init__(opts)

	def get_transforms(self):
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		}
		return transforms_dict


class SketchToImageTransforms(TransformsConfig):

	def __init__(self, opts):
		super(SketchToImageTransforms, self).__init__(opts)

	def get_transforms(self):
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor()]),
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor()]),
		}
		return transforms_dict

# this is added to transform images training/test set used in training the psp decoder for shroud restoration:
class ShroudToImageTransforms(TransformsConfig):
	def __init__(self, opts):
		super(ShroudToImageTransforms, self).__init__(opts) # just save the opts as self.opts.

	# the following was adapted from the corresponding get_transforms methods in SketchToImageTransforms. See above
	def get_transforms(self):
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),

			# this is only for preprocessing. Not needed for training:
			'transform_to_preprocess':transforms.Compose([
				# transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		}
		return transforms_dict




class SegToImageTransforms(TransformsConfig):

	def __init__(self, opts):
		super(SegToImageTransforms, self).__init__(opts)

	def get_transforms(self):
		transforms_dict = {
			# this is used to transform the target (ground truth) images:
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			
			# this is used to downsize the original images,
			# to make it a blurry image.
			# used as the input x in the network during training: 
			'transform_source': transforms.Compose([
				transforms.Resize((256, 256)),
				augmentations.ToOneHot(self.opts.label_nc),
				transforms.ToTensor()]),

			# used in test set. Same as 'transform_gt_train' above'
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((256, 256)),
				augmentations.ToOneHot(self.opts.label_nc),
				transforms.ToTensor()])
		}
		return transforms_dict


class SuperResTransforms(TransformsConfig):

	def __init__(self, opts):
		super(SuperResTransforms, self).__init__(opts)

	def get_transforms(self):
		if self.opts.resize_factors is None:
			self.opts.resize_factors = '1,2,4,8,16,32'
		factors = [int(f) for f in self.opts.resize_factors.split(",")]
		print("Performing down-sampling with factors: {}".format(factors))
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source': transforms.Compose([
				transforms.Resize((256, 256)),
				augmentations.BilinearResize(factors=factors),
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((256, 256)),
				augmentations.BilinearResize(factors=factors),
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		}
		return transforms_dict
