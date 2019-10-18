import os
import sys
import time
import numpy as np
import scipy.misc
import scipy.io as sio

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset

# external modules
from logger import Logger
import pytorch_ssim
from loss_models import PatchImageDiscriminator

# custom modules
from loss import TBNLoss
import tbn_model


class TBNTester:
    def __init__(self, args):
        if args.print_args:
            print('\nArgs:')
            vargs = vars(args)
            for argIdx, argKey in enumerate(vargs, 0):
                print(argKey + ' : ' + str(vargs[argKey]))
            print('\n')
        self.args = args

        if 'gpu' == args.device_mode:
            if not torch.cuda.is_available():
                sys.exit('Error: CUDA was requested but is unavailable.')
            print('using gpu, device: ' + str(args.cuda_device_num))
            self.tensor_type = 'torch.cuda.FloatTensor'

            torch.cuda.set_device(args.cuda_device_num)
            torch.cuda.empty_cache()
        else:
            print('using cpu')
            self.tensor_type = 'torch.FloatTensor'


        if self.args.use_gan:
            if self.args.use_ls_gan:
                self.gan_criterion = nn.MSELoss()
                self.fake_val = -1
            else:
                self.gan_criterion = nn.BCEWithLogitsLoss()
                self.fake_val = 0

            noise_sigma = self.args.gan_noise_sigma if self.args.use_gan_noise else None

            self.discriminator = PatchImageDiscriminator(n_channels=self.args.num_output_channels,
                                                         use_noise=self.args.use_gan_noise,
                                                         noise_sigma=noise_sigma,
                                                         num_intermediate_layers=self.args.gan_num_extra_layers)

        self.out_batch_idx = 0
        self.tensor_write_count = 0

        if self.args.print_output and not os.path.exists(self.args.img_out_dir):
            os.makedirs(self.args.img_out_dir)

        self.num_eval_combine_views = self.args.num_combine_views
        if args.dataset_name == 'chair' or args.dataset_name == 'car':
            import shapenet_img_data_loader as dataset
            args.azim_rotation_angle_increment = 10.0
            args.elev_rotation_angle_increment = 10.0
            args.final_height = 256
            args.final_width = 256
            self.do_run_eval = True
        elif 'drc_' in args.dataset_name:
            import drc_img_data_loader as dataset
            args.azim_rotation_angle_increment = 1.0
            args.elev_rotation_angle_increment = 1.0
            args.final_height = 224
            args.final_width = 224
            self.do_run_eval = False
        else:
            raise ValueError(args.dataset_name)

        self.transform = nn.Upsample(size=[args.final_height - (2 * self.args.crop_y_dim),
                                           args.final_width - (2 * self.args.crop_x_dim)], mode='bilinear')

        shuffle_train = False
        shuffle_test = False
        config_num_input = args.num_combine_views
        _, _, file_test_dataset = \
            dataset.create_default_splits(config_num_input, dataset_name=args.dataset_name,
                                          input_width=args.input_width, input_height=args.input_height,
                                          concat_mask=(4 == args.num_output_channels),
                                          shuffle_train=shuffle_train, shuffle_test=shuffle_test,
                                          img_path=self.args.img_path, args=self.args)

        self.n_file_test_img = file_test_dataset.__len__()
        print('Use a file tuple dataset with', self.n_file_test_img, 'images')
        self.file_test_loader = DataLoader(file_test_dataset,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers, drop_last=False,
                                           shuffle=False)

        if 0.0 < self.args.w_gen_seg3d or self.args.use_seg3d_proxy: self.seg_criterion = nn.MSELoss()

        if 0 == self.args.vol_dim:
            vol_dim = int(self.args.input_width / 2)
            for conv_idx in range(self.args.num_input_convs):
                vol_dim = int(vol_dim / 2)
            print('inferring vol_dim of ' + str(vol_dim))
        else:
            vol_dim = int(self.args.vol_dim)
            print('using vol_dim of ' + str(vol_dim))

        self.num_input_channels = self.args.num_input_channels
        self.num_output_channels = self.args.num_output_channels
        self.device = torch.device((
            ('cuda:' + str(self.args.cuda_device_num)) if 'gpu' == self.args.device_mode else 'cpu'))
        self.loss_function = TBNLoss(tensor_type=self.tensor_type, use_vgg=(0.0 < self.args.w_gen_vgg),
                                     vgg_model_path=self.args.vgg_model_path)
        self.loss_function = self.loss_function.to(self.device)
        self.model = tbn_model.TBN(self.num_input_channels, self.num_output_channels,
                                   args=self.args, vol_dim=vol_dim, num_features=self.args.num_features,
                                   tensor_type=self.tensor_type)

        self.model = self.model.to(self.device)

        if self.args.use_gan:
            self.gan_criterion = self.gan_criterion.to(self.device)
            self.discriminator = self.discriminator.to(self.device)

        if self.tensor_type == 'torch.cuda.FloatTensor':
            torch.cuda.synchronize()

        self.batch_num = 0
        self.test_batch_num = 0
        self.eval_batch_num = 0

        self.total_loss_sum = 0.0
        self.total_test_loss_sum = 0.0
        self.total_eval_loss_sum = 0.0

        self.total_disc_loss_sum = 0.0

        self.logs = self.init_logs()

    @staticmethod
    def ones_like(tensor, device, val=1.):
        return torch.FloatTensor(tensor.size()).fill_(val).to(device)

    def run_eval(self, num_requested_inputs_to_use=0):
        self.eval_batch_num += 1
        self.reset_logs('eval')

        self.model.eval()

        if self.args.use_gan:
            self.discriminator.eval()

        with torch.no_grad():
            test_loss_item = 1e19

            running_loss = 0.0

            start_time = time.time()
            test_loss_sum = 0.0

            num_inputs_to_use = self.num_eval_combine_views if 0 == num_requested_inputs_to_use else num_requested_inputs_to_use

            for (i, test_data) in enumerate(self.file_test_loader, 0):
                if (0) == ((i + 1) % self.args.log_interval):
                    print('start ' + str(i + 1) + ' of ' + str(self.n_file_test_img / self.args.batch_size))

                crnt_batch_size = test_data['tgt_rgb_image'][0].shape[0]

                input_range = 1 if self.args.use_synthetic_input else num_inputs_to_use

                data = self.get_data(test_data, num_inputs_to_use, self.args.use_synthetic_input)

                if self.args.use_synthetic_input:
                    for input_idx in range(1, num_inputs_to_use):
                        # assign pose for image to be generated
                        data['tgt_azim_transform_mode'][0] = data['src_azim_transform_mode'][input_idx]
                        data['tgt_elev_transform_mode'][0] = data['src_elev_transform_mode'][input_idx]

                        model_out = self.model(1, data)

                        data['src_rgb_image'][input_idx] = model_out[0][:, 0:3, :, :]
                        data['src_seg_image'][input_idx] = model_out[2][0]

                    data['tgt_azim_transform_mode'][0] = torch.zeros(data['src_azim_transform_mode'][0].shape).type(
                        'torch.DoubleTensor')
                    data['tgt_elev_transform_mode'][0] = torch.zeros(data['src_elev_transform_mode'][0].shape).type(
                        'torch.DoubleTensor')

                model_out = self.model(num_inputs_to_use, data)
                eval_loss = self.compute_gen_losses(model_out, data, loss_type='eval')

                if self.args.print_output:
                    if self.args.use_seg3d_proxy and self.args.print_occupancy_volume:
                        gen_tgt_occupancy = model_out[4]
                        for idx in range(gen_tgt_occupancy.shape[0]):
                            class_final_bottleneck = gen_tgt_occupancy[idx, :, :, :]

                            class_mat = {}
                            np_class_final_bottleneck = class_final_bottleneck.cpu().detach().numpy()

                            np_class_final_bottleneck = np_class_final_bottleneck.squeeze(0)
                            np_class_final_bottleneck = np.flip(np_class_final_bottleneck, axis=-2)
                            np_class_final_bottleneck = np.swapaxes(np_class_final_bottleneck, 1, 2)

                            class_mat['volume'] = np_class_final_bottleneck

                            sio.savemat(self.args.img_out_dir + '/' + str(self.tensor_write_count + 1) + '.mat', class_mat)

                            self.tensor_write_count += 1

                    src_rgb_image = data['src_rgb_image']
                    tgt_rgb_image = data['tgt_rgb_image'][0]

                    if self.args.use_seg3d_proxy:
                        src_seg_image = data['src_seg_image']
                        tgt_seg_image = data['tgt_seg_image'][0]

                        src_cat_images = None
                        for view_idx in range(0, num_inputs_to_use):
                            if src_cat_images is None:
                                src_cat_images = torch.cat((src_rgb_image[0], torch.cat((src_seg_image[0], src_seg_image[0], src_seg_image[0]), 1)),
                                                           3)
                            else:
                                src_cat_images = torch.cat((src_cat_images, src_rgb_image[view_idx],
                                                            torch.cat((src_seg_image[view_idx], src_seg_image[view_idx], src_seg_image[view_idx]),
                                                                      1)), 3)

                            tgt_seg_rgb = torch.cat((tgt_seg_image, tgt_seg_image, tgt_seg_image), 1)
                            gen_tgt_seg3d = model_out[2][0]
                            gen_tgt_seg3d_rgb = torch.cat((gen_tgt_seg3d, gen_tgt_seg3d, gen_tgt_seg3d), 1)
                            cat_images = torch.cat((src_cat_images[:, 0:3, :, :],
                                                    model_out[0][:, 0:3, :, :], tgt_rgb_image[:, 0:3, :, :],
                                                    gen_tgt_seg3d_rgb[:, 0:3, :, :], tgt_seg_rgb[:, 0:3, :, :]), 3)
                    else:
                        src_cat_images = None
                        for view_idx in range(0, num_inputs_to_use):
                            if src_cat_images is None:
                                src_cat_images = src_rgb_image[0]
                            else:
                                src_cat_images = torch.cat((src_cat_images, src_rgb_image[view_idx]), 3)

                            cat_images = torch.cat((src_cat_images[:, 0:3, :, :],
                                                    model_out[0][:, 0:3, :, :], tgt_rgb_image[:, 0:3, :, :]), 3)

                    for outImgIdx in range(crnt_batch_size):
                        outputFrame = cat_images[outImgIdx, :, :, :]
                        out_str = "%05d" % (self.args.batch_size * self.out_batch_idx + outImgIdx,)
                        scipy.misc.imsave(self.args.img_out_dir + '/' + str(out_str) + '_out.png',
                                          np.squeeze(np.transpose(outputFrame.cpu().detach().numpy(),
                                                                  (1, 2, 0))))

                    self.out_batch_idx = self.out_batch_idx + 1

                if 0 == (i + 1) % self.args.log_interval:
                    crnt_time = time.time()
                    print('end ' + str(i + 1) + ' of ' + str(self.n_file_test_img / self.args.batch_size))
                    print(
                        'time:',
                        round(crnt_time - start_time, 3),
                        's',
                        'SSIM Loss:',
                        self.logs['l_eval_gen_raw_ssim'].item() / (i + 1),
                        'L1:',
                        self.logs['l_eval_gen_raw_l1'].item() / (i + 1),
                        'Final SSIM:',
                        1 - self.logs['l_eval_gen_raw_ssim'].item() / (i + 1),
                    )
                    start_time = crnt_time
                    test_loss_sum = 0.0

        final_scale_factor = float(self.n_file_test_img) / self.args.batch_size

        self.logs['l_eval_gen'] /= final_scale_factor
        self.logs['l_eval_gen_gan'] /= final_scale_factor
        self.logs['l_eval_gen_l1'] /= final_scale_factor
        self.logs['l_eval_gen_raw_l1'] /= final_scale_factor
        self.logs['l_eval_gen_raw_ssim'] /= final_scale_factor
        self.logs['l_eval_gen_ssim'] /= final_scale_factor
        self.logs['l_eval_gen_seg3d'] /= final_scale_factor
        self.logs['l_eval_gen_vgg'] /= final_scale_factor
        self.logs['l_eval_gen_running'] /= final_scale_factor

        eval_gen_vgg = self.logs['l_eval_gen_vgg'] / self.args.w_gen_vgg if 0.0 < self.args.w_gen_vgg else 0.0

        print(
            'Eval tuples test:',
            'SSIM:',
            1.0 - self.logs['l_eval_gen_raw_ssim'].item(),
            'L1:',
            self.logs['l_eval_gen_raw_l1'].item(),
        )

        running_loss /= self.n_file_test_img / self.args.batch_size

        return running_loss, (1.0 - self.logs['l_eval_gen_raw_ssim'])

    @staticmethod
    def init_logs():
        return {'l_eval_gen': 0.0,
                'l_eval_gen_gan': 0.0,
                'l_eval_gen_l1': 0.0,
                'l_eval_gen_raw_l1': 0.0,
                'l_eval_gen_raw_ssim': 0.0,
                'l_eval_gen_ssim': 0.0,
                'l_eval_gen_seg3d': 0.0,
                'l_eval_gen_vgg': 0.0,
                'l_eval_gen_running': 0.0}

    def reset_logs(self, log_type='train'):
        self.logs['l_' + log_type + '_gen'] = 0.0
        self.logs['l_' + log_type + '_gen_gan'] = 0.0
        self.logs['l_' + log_type + '_gen_l1'] = 0.0
        self.logs['l_' + log_type + '_gen_raw_l1'] = 0.0
        self.logs['l_' + log_type + '_gen_raw_ssim'] = 0.0
        self.logs['l_' + log_type + '_gen_ssim'] = 0.0
        self.logs['l_' + log_type + '_gen_seg3d'] = 0.0
        self.logs['l_' + log_type + '_gen_vgg'] = 0.0
        self.logs['l_' + log_type + '_gen_running'] = 0.0
        if 'train' == log_type:
            self.logs['l_' + log_type + '_disc'] = 0.0
            self.logs['l_' + log_type + '_disc_gan'] = 0.0
            self.logs['l_' + log_type + '_disc_running'] = 0.0

    def compute_gen_losses(self, model_out, data, loss_type='train'):
        loss = 0.0

        tgt_rgb_image = data['tgt_rgb_image'][0]
        tgt_seg_image = data['tgt_seg_image'][0]
        if self.args.upsample_output:
            orig_tgt_rgb_image = data['orig_tgt_rgb_image'][0]
            orig_tgt_seg_image = data['orig_tgt_seg_image'][0]

        src_seg_image = data['src_seg_image']

        if self.args.upsample_output:
            upsample_model_out = []
            upsample_model_out.append(self.transform(model_out[0]))
            upsample_model_out.append(model_out[1])
            if self.args.use_seg3d_proxy:
                upsample_model_out.append([self.transform(model_out[2][0])])
            else:
                upsample_model_out.append(model_out[2])

            loss_gen = upsample_model_out
            loss_tgt_rgb_image = orig_tgt_rgb_image
            loss_tgt_seg_image = orig_tgt_seg_image

            gen_src_seg3d = upsample_model_out[1]
            gen_tgt_seg3d = upsample_model_out[2]
        else:
            loss_gen = model_out
            loss_tgt_rgb_image = tgt_rgb_image
            loss_tgt_seg_image = tgt_seg_image

            gen_src_seg3d = model_out[1]
            gen_tgt_seg3d = model_out[2]

        raw_vgg_loss, raw_l1_loss, raw_ssim_loss = self.loss_function(loss_gen[0], loss_tgt_rgb_image)

        raw_seg3d_loss = torch.zeros(raw_l1_loss.shape).to(self.device)
        if self.args.use_seg3d_proxy:
            num_src_imgs = len(gen_src_seg3d)
            gen_src_seg = model_out[3]
            for view_idx in range(0, num_src_imgs):
                raw_seg3d_loss += 0.5 * self.seg_criterion(gen_src_seg3d[view_idx], src_seg_image[view_idx])
                raw_seg3d_loss += 0.5 * self.seg_criterion(gen_src_seg[view_idx], src_seg_image[view_idx])
            if 0 < len(gen_tgt_seg3d):
                if self.args.upsample_output:
                    gen_tgt_seg = self.transform(model_out[0][:, 3:4, :, :])
                else:
                    gen_tgt_seg = model_out[0][:, 3:4, :, :]
                raw_seg3d_loss += 0.5 * self.seg_criterion(gen_tgt_seg3d[0], loss_tgt_seg_image)
                raw_seg3d_loss += 0.5 * self.seg_criterion(gen_tgt_seg, loss_tgt_seg_image)
                num_src_imgs += 1
            raw_seg3d_loss /= num_src_imgs

        l_gen_vgg_loss = self.args.w_gen_vgg * raw_vgg_loss.mean()
        loss += l_gen_vgg_loss
        l_gen_l1_loss = self.args.w_gen_l1 * raw_l1_loss.mean()
        loss += l_gen_l1_loss

        if self.args.normalize_ssim_loss:
            nonzero_ssim_loss = (raw_ssim_loss + 1.0)
            normalized_ssim_loss = 0.5 * nonzero_ssim_loss
            l_log_ssim_loss_val = (2.0 - nonzero_ssim_loss.mean())
            ssim_loss = (2.0 - nonzero_ssim_loss)
        else:
            ssim_loss = raw_ssim_loss
            ssim_loss = (1.0 - ssim_loss)
            l_log_ssim_loss_val = ssim_loss.mean()

        l_gen_ssim_loss = self.args.w_gen_ssim * ssim_loss.mean()
        loss += l_gen_ssim_loss

        l_gen_seg3d_loss = self.args.w_gen_seg3d * raw_seg3d_loss.mean()
        loss += l_gen_seg3d_loss

        if self.args.use_gan and 0.0 < self.args.w_gen_gan_label:
            fake_labels, _ = self.discriminator(model_out[0])

            self.ones = self.ones_like(fake_labels, device=self.device)

            l_gen_gan_loss = self.args.w_gen_gan_label * self.gan_criterion(fake_labels, self.ones).mean()
            loss += l_gen_gan_loss

            self.logs['l_' + loss_type + '_gen_gan'] += l_gen_gan_loss.item()
        self.logs['l_' + loss_type + '_gen_l1'] += l_gen_l1_loss.item()
        self.logs['l_' + loss_type + '_gen_ssim'] += l_gen_ssim_loss.item()
        self.logs['l_' + loss_type + '_gen_seg3d'] += l_gen_seg3d_loss.item()
        self.logs['l_' + loss_type + '_gen_vgg'] += l_gen_vgg_loss.item()

        self.logs['l_' + loss_type + '_gen_raw_l1'] += raw_l1_loss.mean()
        self.logs['l_' + loss_type + '_gen_raw_ssim'] += l_log_ssim_loss_val

        self.logs['l_' + loss_type + '_gen'] += loss.item()

        if 'train' == loss_type:
            self.total_loss_sum += loss.item()
            running_loss = self.total_loss_sum / self.batch_num
        elif 'test' == loss_type:
            self.total_test_loss_sum += loss.item()
            running_loss = self.total_test_loss_sum / self.test_batch_num
        elif 'eval' == loss_type:
            self.total_eval_loss_sum += loss.item()
            running_loss = self.total_eval_loss_sum / self.eval_batch_num
        else:
            raise ValueError(loss_type)

        self.logs['l_' + loss_type + '_gen_running'] += running_loss
        return loss

    def get_data(self, data, num_inputs_to_use=1, use_synthetic_input=False, num_outputs_to_use=1):
        output_data = data

        for output_idx in range(num_outputs_to_use):
            output_data['tgt_rgb_image'][output_idx] = data['tgt_rgb_image'][output_idx].to(self.device)
            output_data['tgt_seg_image'][output_idx] = data['tgt_seg_image'][output_idx].to(self.device)
            if self.args.upsample_output:
                output_data['orig_tgt_rgb_image'][output_idx] = data['orig_tgt_rgb_image'][output_idx].to(self.device)
                output_data['orig_tgt_seg_image'][output_idx] = data['orig_tgt_seg_image'][output_idx].to(self.device)

        for input_idx in range(num_inputs_to_use):
            output_data['src_rgb_image'][input_idx] = data['src_rgb_image'][input_idx][:, 0:3, :, :].to(self.device)
            output_data['src_seg_image'][input_idx] = data['src_seg_image'][input_idx].to(self.device)

            if use_synthetic_input and not self.args.use_random_transforms:
                if 0 != input_idx:
                    # regularly sample positions around the central axis
                    angle = (input_idx - 1) * (360.0 / (num_inputs_to_use - 1))
                    output_data['src_azim_transform_mode'][input_idx] = angle * torch.ones(
                        data['src_azim_transform_mode'][input_idx].shape).type('torch.DoubleTensor')
                    output_data['src_elev_transform_mode'][input_idx] = torch.zeros(
                        data['src_elev_transform_mode'][input_idx].shape).type('torch.DoubleTensor')
                else:
                    output_data['src_azim_transform_mode'][input_idx] = data['src_azim_transform_mode'][input_idx].type(
                        'torch.DoubleTensor')
                    output_data['src_elev_transform_mode'][input_idx] = data['src_elev_transform_mode'][input_idx].type(
                        'torch.DoubleTensor')

        return output_data

    def load(self, path, load_disc=True, in_disc_path=''):
        if os.path.exists(path):
            if 'cpu' == self.args.device_mode:
                self.model = torch.load(path, map_location='cpu')
            else:
                self.model = torch.load(path)
            if isinstance(self.model, torch.nn.DataParallel):
                self.model = self.model.module
            self.model.tensor_type = self.tensor_type
            self.model.args = self.args
            self.model = self.model.to(self.device)
        else:
            print('generator file not found: ' + path)
            exit(-1)

        if self.args.use_gan and load_disc:
            disc_path = path[:-4] + '_disc.pth' if '' == in_disc_path else in_disc_path

            if os.path.exists(disc_path):
                self.discriminator = torch.load(disc_path)
                if isinstance(self.discriminator, torch.nn.DataParallel):
                    self.discriminator = self.discriminator.module
                self.discriminator = self.discriminator.to(self.device)
            else:
                print('discriminator file not found: ' + disc_path)
