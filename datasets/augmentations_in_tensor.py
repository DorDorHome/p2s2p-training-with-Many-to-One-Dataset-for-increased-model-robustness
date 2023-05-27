# this file was created by Alvin, to implement the class in augmentations.py
# directly on pytorch.tensors.
# the purpose is to create transformation classes that does the same thing
# but can be applied to datasets.augmentations_in_tensor.py

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms


class BilinearResize_in_tensor(object):
    """
    this acts on tensors.
    Almost the same as the class with the same name in datasets/augmentations.py
    
    
    """
    def __init__(self, factors=[1, 2, 4, 8, 16, 32]):
        self.factors = factors # when called, usually called with a list of only one member.

    def __call__(self, image):
	    factor = np.random.choice(self.factors, size=1)[0]
	    D = BicubicDownSample(factor=factor, cuda=False)
	    # img_tensor = transforms.ToTensor()(image).unsqueeze(0) # create an extra dimension on 0 axis
	    img_tensor_lr = D(image)[0].clamp(0, 1) # .unsqueeze(0)
		# img_low_res = transforms.ToPILImage()(img_tensor_lr)
	    return img_tensor_lr










#  BicubicDownSample:
# this acts on tensor so shouldn't need any changes (??)

class BicubicDownSample(nn.Module):
	def bicubic_kernel(self, x, a=-0.50):
		"""
		This equation is exactly copied from the website below:
		https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bicubic
		"""
		abs_x = torch.abs(x)
		if abs_x <= 1.:
			return (a + 2.) * torch.pow(abs_x, 3.) - (a + 3.) * torch.pow(abs_x, 2.) + 1
		elif 1. < abs_x < 2.:
			return a * torch.pow(abs_x, 3) - 5. * a * torch.pow(abs_x, 2.) + 8. * a * abs_x - 4. * a
		else:
			return 0.0

	def __init__(self, factor=4, cuda=True, padding='reflect'):
		super().__init__()
		self.factor = factor
		size = factor * 4
		k = torch.tensor([self.bicubic_kernel((i - torch.floor(torch.tensor(size / 2)) + 0.5) / factor)
						  for i in range(size)], dtype=torch.float32)
		k = k / torch.sum(k)
		k1 = torch.reshape(k, shape=(1, 1, size, 1))
		self.k1 = torch.cat([k1, k1, k1], dim=0)
		k2 = torch.reshape(k, shape=(1, 1, 1, size))
		self.k2 = torch.cat([k2, k2, k2], dim=0)
		self.cuda = '.cuda' if cuda else ''
		self.padding = padding
		for param in self.parameters():
			param.requires_grad = False

	def forward(self, x, nhwc=False, clip_round=False, byte_output=False):
		filter_height = self.factor * 4
		filter_width = self.factor * 4
		stride = self.factor

		pad_along_height = max(filter_height - stride, 0)
		pad_along_width = max(filter_width - stride, 0)
		filters1 = self.k1.type('torch{}.FloatTensor'.format(self.cuda))
		filters2 = self.k2.type('torch{}.FloatTensor'.format(self.cuda))

		# compute actual padding values for each side
		pad_top = pad_along_height // 2
		pad_bottom = pad_along_height - pad_top
		pad_left = pad_along_width // 2
		pad_right = pad_along_width - pad_left

		# apply mirror padding
		if nhwc:
			x = torch.transpose(torch.transpose(x, 2, 3), 1, 2)   # NHWC to NCHW

		# downscaling performed by 1-d convolution
		x = F.pad(x, (0, 0, pad_top, pad_bottom), self.padding)
		x = F.conv2d(input=x, weight=filters1, stride=(stride, 1), groups=3)
		if clip_round:
			x = torch.clamp(torch.round(x), 0.0, 255.)

		x = F.pad(x, (pad_left, pad_right, 0, 0), self.padding)
		x = F.conv2d(input=x, weight=filters2, stride=(1, stride), groups=3)
		if clip_round:
			x = torch.clamp(torch.round(x), 0.0, 255.)

		if nhwc:
			x = torch.transpose(torch.transpose(x, 1, 3), 1, 2)
		if byte_output:
			return x.type('torch.ByteTensor'.format(self.cuda))
		else:
			return x

