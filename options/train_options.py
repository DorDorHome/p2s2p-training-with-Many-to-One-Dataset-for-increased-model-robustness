# Alvin: created a new PreprocessOptions class, for preprocessing data

from argparse import ArgumentParser
from configs.paths_config import model_paths


class TrainOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--use_many_to_one_dataset',
					default=False,
					type = bool,
					help = '')


		self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
		self.parser.add_argument('--dataset_type', default='ffhq_encode', type=str, help='Type of dataset/experiment to run')
		self.parser.add_argument('--encoder_type', default='GradualStyleEncoder', type=str, help='Which encoder to use')
		self.parser.add_argument('--input_nc', default=3, type=int, help='Number of input image channels to the psp encoder')
		self.parser.add_argument('--label_nc', default=0, type=int, help='Number of input label channels to the psp encoder')
		self.parser.add_argument('--output_size', default=1024, type=int, help='Output size of generator')

		self.parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
		self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
		self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')

		self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
		self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
		self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
		self.parser.add_argument('--start_from_latent_avg', action='store_true', help='Whether to add average latent vector to generate codes from encoder.')
		self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space instead of w+')

		self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
		self.parser.add_argument('--id_lambda', default=0, type=float, help='ID loss multiplier factor')
		self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
		self.parser.add_argument('--w_norm_lambda', default=0, type=float, help='W-norm loss multiplier factor')
		self.parser.add_argument('--lpips_lambda_crop', default=0, type=float, help='LPIPS loss multiplier factor for inner image region')
		self.parser.add_argument('--l2_lambda_crop', default=0, type=float, help='L2 loss multiplier factor for inner image region')
		self.parser.add_argument('--moco_lambda', default=0, type=float, help='Moco-based feature similarity loss multiplier factor')

		self.parser.add_argument('--stylegan_weights', default=model_paths['stylegan_ffhq'], type=str, help='Path to StyleGAN model weights')
		self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to pSp model checkpoint')

		self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps')
		self.parser.add_argument('--image_interval', default=100, type=int, help='Interval for logging train images during training')
		self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
		self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
		self.parser.add_argument('--save_interval', default=None, type=int, help='Model checkpoint interval')

		# arguments for weights & biases support
		self.parser.add_argument('--use_wandb', action="store_true", help='Whether to use Weights & Biases to track experiment.')

		# arguments for super-resolution
		self.parser.add_argument('--resize_factors', type=str, default=None, help='For super-res, comma-separated resize factors to use for inference.')

		# arguments for using pretrained psp models:
		self.parser.add_argument('--pretrained_psp_checkpoint_path', type = str, default = None, help = 'if specified, will load the specified pretrained psp prior to training.')


	def parse(self):
		opts = self.parser.parse_args()
		return opts


class PreprocessOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# # add the arguments:
		# self.parser.add_argument('input_folder',
		# 					type = str,
		# 					help = 'The folder that contains the images to be split'
		# 					)

		# self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')

		# self.parser.add_argument('--out_path_for_training_source', type=str, help='Path to the output of the preprocessing for source training data' )
		# self.parser.add_argument('--out_path_for_training_target', type=str, help='Path to the output of the preprocessing for target training data' )
		self.parser.add_argument('--right_half_result',
						type = str,
							help = 'The folder to contain the right hand side of the split images'
							) 
						

		self.parser.add_argument('--left_half_result',
						type = str,
							help = 'The folder to contain the left hand side of the split images'
							) 

		self.parser.add_argument('--dataset_type', default='shroud_to_image', type=str, help='Type of dataset/experiment to run')

		self.parser.add_argument('--label_nc', default=0, type=int, help='Number of input label channels to the psp encoder')
		self.parser.add_argument('--input_nc', default=3, type=int, help='Number of input image channels to the psp encoder')
		#self.parser.add_argument('--label_nc', default=0, type=int, help='Number of input label channels to the psp encoder')


	def parse(self):
		opts = self.parser.parse_args()
		return opts

class PreprocessJesusOptions:
	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
				# the following are supposed to be the same as the training set used:
		self.parser.add_argument('--label_nc', default=0, type=int, help='Number of input label channels to the psp encoder')
		self.parser.add_argument('--input_nc', default=3, type=int, help='Number of input image channels to the psp encoder')


		# the following arguments have default settings that aren't supposed to be changed for normal use:

		self.parser.add_argument('--dataset_type', 
								default='original_shroud',
								type = str,
								help = 'key values for the data_configs.DATASET dict')

		# arguments to specify the location of the input and output folder
		# have no influence on the manipulation used:
		self.parser.add_argument('--original_shroud_input_folder',
									type = str
									)

		self.parser.add_argument('--transformed_shroud_output_folder',
							type = str
							)

		# Arguments that are supposed to be changed, to experiment with different levels of manipulations:
		self.parser.add_argument('--Gaussian_blur_kernel_size',
								type=float,
								help = 'kernel size to be used in the Gaussian Blur')
		self.parser.add_argument("--Gaussian_blur_sigma",
						type=float,
						help ='blur std in the Gaussian blur to be applied to the original shroud image.' )

		# denoise: used if Gaussian blur do not produce good results.
		self.parser.add_argument('--denoise_amount',
						default= 0,
						type = float,
						help = 'denoise amount to be applied to the original shroud image.')
		self.parser.add_argument('--convert_to_BW',
							default=False,
							type = bool,
							help = 'whether to convert the original shroud image to B&W.')

	def parse(self):
		opts = self.parser.parse_args()
		return opts