# this file is created by Alvin
# by adapting the inference.py file

# the purpose is to improve the super-resolution results by updating the weights of stylegan
# dependency: need an additional file, downsize_photos.py


# step 1: inference using pretrained psp_celebs_super_resolution model
# step 2: train using L_2 norm:
# - downsize the result back to the original size (i.e. make it blurry)
# - loss = L_2 between original blurry image and new images
# - optionally, also use adversarial loss 
# - print the result for every 500 iterations
# - save the weights




# importing the same library as in the scripts/inference.py :
import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp

# additional downsize tools created by Alvin Chan:
# from added_data_preprocessing_tools.downsize_photos import downsize_image_tensor
import torchvision.transforms as transforms

# for downsizing in tensor space:
from datasets import augmentations

# for defining loss function:
import torch.nn.functional as F


# for saving the output as png: possibly not required later:
from torchvision import utils


# function used to freeze the parameters to the untrainable (used in step 3, )
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag




# set device to be used:

device = 'cuda'

# step 1: inference using pretrained model:

def run():
    test_opts = TestOptions().parse()

    # print(type(test_opts))
    # print(test_opts.checkpoint_path)

    # Create folder for output:
    # by default, this is set to in a folder named "path_to_experiment/superresolution_with_fine_tuning"
    if test_opts.resize_factors is not None:
        assert len(
            test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results',
                                        'downsampling_{}'.format(test_opts.resize_factors))
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled',
                                        'downsampling_{}'.format(test_opts.resize_factors))
    else:
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location=device)
    # check the type of ckpt:
    # print(f'type of checkpoint: {type(ckpt)}')

    # setting up the opts:
    opts = ckpt['opts']
    print('\n before updating with test_opts, opts is: ', opts)

    # 
    opts.update(vars(test_opts))

    print('\n opts after updating with test_opts:', opts)
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)

    # loading the model:
    net = pSp(opts)
    net.eval()
    net.cuda()


    #testing for the deployment of the decoder:
    # this is done by setting input_code to be true.


    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()

    print('loading photos to be downsized from path: ', opts.data_path)

    dataset = InferenceDataset(root=opts.data_path,
                                transform=transforms_dict['transform_inference'], # when doing superresolution, this transform downsize the input image
                                opts=opts) # this determines, among other things, the scale of the downsizing.
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    print('size of dataset; ', len(dataset))
    print(type(dataset[0]))
    print(dataset[1].size())

    # showing the first downsized photo:
    # print('\n showing the first image in the dataset (downsizing of the original was applied)')
    # tensor2im(dataset[0]).show()

    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    global_time = []
    batch_number = 0
    for input_batch in tqdm(dataloader):# loading the images by batches
        batch_number +=1 
        # stop when desired of images is reached:
        if global_i >= opts.n_images:
            break

        # forward pass, without needed to backprop
        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            tic = time.time()
            print('opts.latent_mask is: ', opts.latent_mask) # expect 'None' for super-res
            print('opts.resize_outputs', opts.resize_outputs)
            # running on batch using the function below. Need to be replaced:
            #result_batch = run_on_batch(input_cuda, net, opts)




        # for each image in each_batch:

        for i in range(opts.test_batch_size):
            
            # check whether the tensor requires grad:
            print(f'downsized image batch at {i} image size: {input_cuda[i].size()}')
            print(f'downsized image batch at {i} requires grad{input_cuda[i].requires_grad}') # expect False
            # Main Steps:
            # Step 1:
            # run psp encoder for image at position i:
            # save the resultant latent vector

            __, result_latent = net(input_cuda[i].unsqueeze(0), randomize_noise = False,
             resize = opts.resize_outputs, 
             return_latents = True
             )

            print('shape of result latent:', result_latent.shape)



            # Step 2:
            # forward pass this latent
            g_ema = net.decoder #extract the decoder part (stylegan) refer to psp_stylegan2_plaground
            with torch.no_grad():
                g_ema.eval()
                one_sample_output, _ = g_ema([result_latent], input_is_latent = True)
            
            print('one_sample shape', one_sample_output.shape)

            one_sample_output_path = os.path.join(out_path_results, f'{batch_number}th_bath_{i}th_output_image_before_optimization.png')
            # saving the output:
            utils.save_image(
                one_sample_output,
                one_sample_output_path,
                nrow = 1,
                normalize = True,
                range = (-1,1)

            )

             # show the result, to check the breakdown function works the same way:
            # print('decoded output image type: ', type(one_sample))
            tensor2im(one_sample_output.squeeze(0)).show()

            factors = [int(f) for f in opts.resize_factors.split(",")]
            print('step 3: resize factors: ', factors)

            # This needs to be changed,
            # because augmentations.BilinearResize acts on PIl images
            # so backprop cannot be done.
            # See the file data
            blur_transform_in_tensor = transforms.Compose([
                                            transforms.Normalize([0,0,0], [2,2,2]),
                                            transforms.Normalize([-0.5, -0.5, 0.5], [1,1,1]),
                                            transforms.ToPILImage(),
                                            transforms.Resize((256, 256)),
                                            augmentations.BilinearResize(factors=factors),
                                            transforms.Resize((256, 256)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                            ])
            

            # Step 3: Set the Output of the first half (the psp encoder) to be trainable

            # check whether result_latent is trainable:

            # latent_to_optimize = result_latent.clone().detach().to(device).requires_grad_(True)
            latent_to_optimize = torch.autograd.Variable(result_latent, requires_grad = True)
            print('result_latent requires grad??: ', latent_to_optimize.requires_grad)

            # set the parameters of g_ema to be untrainable:
            requires_grad(g_ema, False)

            # step 4: set up the optimizer for the optimization:
            learning_rate = opts.learning_rate
            optimizer_for_w_space = torch.optim.Adam([latent_to_optimize], lr = learning_rate)

            # (for debugging): try downsizing the output image (checked)
            # one_sample_output_downsized = blur_transform_in_tensor(one_sample_output.squeeze(0))
            # # show the downsized output:
            # tensor2im(one_sample_output_downsized).show()

            #debug: check tensor device:
            print('location of latent to optimize: ', latent_to_optimize.device)
            g_ema.eval()
            g_ema.cuda()
            #print('location of g_ema: ', g_ema.device)

            # step 5: start optimizing the w for output a
            pbar_w_training = tqdm(range(opts.n_epoch))
            for w_opt_epoch in pbar_w_training:
                print(f'\n starting optimization on image {i} in batch {batch_number}')
                print('result_latent requires grad??: ', latent_to_optimize.requires_grad)
                output_img, _ = g_ema([latent_to_optimize], input_is_latent = True)
                output_img_downsized = blur_transform_in_tensor(output_img.squeeze(0)).to(device)
                print('output_img shape: ', output_img.shape)
                print('whether output img requires grad', output_img.requires_grad)
                print('img downsized shape: ', output_img_downsized.shape)
                print('whether output img downsized requires grad: ', output_img_downsized.requires_grad)
                print('input cuda (original blurred input image) shape:', input_cuda[i].squeeze(0).shape)

                tensor2im(input_cuda[i].squeeze(0)).show()
                # loss:
                loss = F.mse_loss(output_img_downsized, input_cuda[i].unsqueeze(0))
                print('\n loss is ', loss)
                
                #get gradient:
                loss.backward()
                optimizer_for_w_space.step()
                optimizer_for_w_space.zero_grad()
                if epoch % 200 == 0:
                    print('saving epoch {}'.format(w_opt_epoch))
                    output_img_after_optimized_path = os.path.join(one_sample_output_path, f'batch_{batch_number}_image{i}_optimized_for_{w_opt_epoch}_epoch.png')
                    utils.save(
                        output_img,
                        output_img_after_optimized_path
                    )

            print('\n finished training! ')
            final_output_img_after_optimized_path = os.path.join(one_sample_output_path, f'batch_{batch_number}_image{i}_optimized_for_{opt.n_epoch}_epoch_lr_{learning_rate}.png')
            final_output_img, _ = g_ema([latent_to_optimize],input_is_latent = True)
            utils.save(
                final_output_img.
                final_output_img_after_optimized_path
            )








            # for debugging only, for the result:
            # print(f'result image batch at {i} image size: {result_image.size()}')
            # print(f'result image batch at {i} requires grad{result_image.requires_grad}') # expect False

            
            # toc = time.time()
            # global_time.append(toc - tic)

            # # possibly useful: 
            # result = tensor2im(result_batch[i])
            # im_path = dataset.paths[global_i]


    #         if opts.couple_outputs or global_i % 100 == 0:
    #             input_im = log_input_image(input_batch[i], opts)
    #             resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
    #             if opts.resize_factors is not None:
    #                 # for super resolution, save the original, down-sampled, and output
    #                 source = Image.open(im_path)
    #                 res = np.concatenate([np.array(source.resize(resize_amount)),
    #                                         np.array(input_im.resize(resize_amount, resample=Image.NEAREST)),
    #                                         np.array(result.resize(resize_amount))], axis=1)
    #             else:
    #                 # otherwise, save the original and output
    #                 res = np.concatenate([np.array(input_im.resize(resize_amount)),
    #                                         np.array(result.resize(resize_amount))], axis=1)
    #             Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

    #         im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
    #         Image.fromarray(np.array(result)).save(im_save_path)

    #         global_i += 1

    # stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    # result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    # print(result_str)

    # with open(stats_path, 'w') as f:
    #     f.write(result_str)


def run_on_batch(inputs, net, opts):
    if opts.latent_mask is None:
        result_batch = net(inputs, randomize_noise=False, resize=opts.resize_outputs)
    else:
        latent_mask = [int(l) for l in opts.latent_mask.split(",")]
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
            # get output image with injected style vector
            res = net(input_image.unsqueeze(0).to("cuda").float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject,
                      alpha=opts.mix_alpha,
                      resize=opts.resize_outputs)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch

if __name__ == '__main__':
    run()

