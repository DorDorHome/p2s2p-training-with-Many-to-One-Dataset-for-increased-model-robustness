from configs import transforms_config
from configs.paths_config import dataset_paths

# added for debug:
# import transforms_config
# import paths_config.dataset_paths

DATASETS = {
	'ffhq_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['ffhq'], 
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'ffhq_frontalize': {
		'transforms': transforms_config.FrontalizationTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'celebs_sketch_to_face': {
		'transforms': transforms_config.SketchToImageTransforms,
		'train_source_root': dataset_paths['celeba_train_sketch'],
                                               
	},
	'celebs_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'train_source_root': dataset_paths['celeba_train_segmentation'],
		'train_target_root': dataset_paths['celeba_train'],
		'test_source_root': dataset_paths['celeba_test_segmentation'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'celebs_super_resolution': {
		'transforms': transforms_config.SuperResTransforms,
		'train_source_root': dataset_paths['celeba_train'],
		'train_target_root': dataset_paths['celeba_train'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'shroud_to_image':{
		'transforms': transforms_config.ShroudToImageTransforms,
		'train_source_root': dataset_paths['shroud_train_source'],
		'train_target_root': dataset_paths['shroud_train_target'],
		'test_source_root': dataset_paths['shroud_test_source'],
		'test_target_root': dataset_paths['shroud_test_target'],
		'to_preprocess_train': dataset_paths['to_preprocess_train'],
		'to_preprocess_test': dataset_paths['to_preprocess_test']
	},
	'original_shroud':{
		'transforms':transforms_config.TransformOriginalShroud,
		# 'original_shroud_input_folder':dataset_paths['original_shroud_input_folder'],
		# 'transformed_shroud_output_folder':dataset_paths['transformed_shroud_output_folder']
	}

	}


print('debugging:, items of dataset', DATASETS['ffhq_encode']['train_source_root'])