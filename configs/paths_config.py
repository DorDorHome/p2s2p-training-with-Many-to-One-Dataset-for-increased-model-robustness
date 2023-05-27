
dataset_paths = {
	'celeba_train': '',
	'celeba_test': '/home/sfchan/Desktop/Datasets/faces/celeba/celeba/images_aligned',
	'celeba_train_sketch': '',
	'celeba_test_sketch': '',
	'celeba_train_segmentation': '',
	'celeba_test_segmentation': '',
	'ffhq': '/home/sfchan/Desktop/Models/collections of image generation model using GAN/stylegan2-pytorch-master/dataset/ffhq',
	'shroud_train_source':  "/home/sfchan/Desktop/Datasets/faces/shroud/shroud_train/source/" ,
	'shroud_train_target': "/home/sfchan/Desktop/Datasets/faces/shroud/shroud_train/target/",
	'shroud_test_source':"/home/sfchan/Desktop/Datasets/faces/shroud/shroud_test/source/",
	'shroud_test_target': "/home/sfchan/Desktop/Datasets/faces/shroud/shroud_test/target/",
	'to_preprocess_train':'/media/sfchan/Expansion/Datasets/dataset_for_shroud_project/datasets/Celeb2Shroud/train/',
	'to_preprocess_test':'/media/sfchan/Expansion/Datasets/dataset_for_shroud_project/datasets/Celeb2Shroud/test/test'
}

model_paths = {
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pth.tar'
}

print(dataset_paths['ffhq'])
print(dataset_paths['celeba_test'])
print(dataset_paths.items())