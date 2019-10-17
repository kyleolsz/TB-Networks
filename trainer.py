import os
import sys
import time
import numpy as np
import scipy.misc
import scipy.io as sio

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

# external modules
from logger import Logger
import pytorch_ssim
from loss_models import PatchImageDiscriminator

# custom modules
from loss import TBNLoss
import tbn_model


class TBNTrainer:
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

        if self.args.use_amp:
            self.adam_betas = (0.9, 0.99)
            self.adam_eps = 1e-04
        else:
            self.adam_betas = (0.9, 0.999)
            self.adam_eps = 1e-08

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

        if 0 != self.args.log_interval and not os.path.exists(self.args.log_folder):
            os.makedirs(self.args.log_folder)

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
        train_dataset, test_dataset, file_test_dataset = \
            dataset.create_default_splits(config_num_input, dataset_name=args.dataset_name,
                                          input_width=args.input_width, input_height=args.input_height,
                                          concat_mask=(4 == args.num_output_channels),
                                          shuffle_train=shuffle_train, shuffle_test=shuffle_test,
                                          img_path=self.args.img_path, args=self.args)

        shuffle = True
        self.n_img = train_dataset.__len__()
        print('Use a train dataset with', self.n_img, 'images')
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers, drop_last=True,
                                       shuffle=shuffle)

        self.n_test_img = test_dataset.__len__()
        print('Use a test dataset with', self.n_test_img, 'images')
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers, drop_last=True,
                                      shuffle=shuffle)

        self.n_file_test_img = file_test_dataset.__len__()
        print('Use a file tuple dataset with', self.n_file_test_img, 'images')
        self.file_test_loader = DataLoader(file_test_dataset,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers, drop_last=False,
                                           shuffle=False)

        if 0 != self.args.test_interval: self.test_loader_iter = iter(self.test_loader)
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

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate,
                                    betas=self.adam_betas, eps=self.adam_eps)
        if self.args.use_gan:
            self.gan_criterion = self.gan_criterion.to(self.device)
            self.discriminator = self.discriminator.to(self.device)
            self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=args.disc_learning_rate,
                                             betas=self.adam_betas, eps=self.adam_eps)

        if self.tensor_type == 'torch.cuda.FloatTensor':
            torch.cuda.synchronize()

        self.batch_num = 0
        self.test_batch_num = 0
        self.eval_batch_num = 0

        self.total_loss_sum = 0.0
        self.total_test_loss_sum = 0.0
        self.total_eval_loss_sum = 0.0

        self.total_disc_loss_sum = 0.0

        self.logger = Logger(self.args.log_folder)
        self.logs = self.init_logs()

    @staticmethod
    def ones_like(tensor, device, val=1.):
        return torch.FloatTensor(tensor.size()).fill_(val).to(device)

    @staticmethod
    def zeros_like(tensor, device, val=0.):
        return torch.FloatTensor(tensor.size()).fill_(val).to(device)

    @staticmethod
    def images_to_numpy(tensor):
        generated = tensor.data.cpu().numpy().transpose(0, 2, 3, 1)
        generated[generated < 0] = 0
        generated[generated > 1] = 1
        generated = generated * 255
        return generated.astype('uint8')

    @staticmethod
    def adjust_learning_rate(optimizer, epoch, learning_rate=0.0002):
        lr = learning_rate

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        self.best_epoch_raw_ssim_loss = -1e19

        had_training_error = False

        if self.args.use_amp:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this program.")
            print('enabling AMP...')
            opt_level = "O1"
            amp.register_float_function(torch, 'batch_norm')
            if self.args.use_gan:
                self.discriminator, self.disc_optimizer = amp.initialize(self.discriminator, self.disc_optimizer, opt_level=opt_level, loss_scale=1.0)
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=opt_level, loss_scale=1.0)
            # disable use of ssim loss for training due to issues with mixed precision
            self.args.w_gen_ssim = 0.0

        if self.args.use_data_parallel:
            print('enabling data parallel...')
            self.model = nn.DataParallel(self.model)
            self.model = self.model.to(self.device)
            self.loss_function = nn.DataParallel(self.loss_function)
            self.loss_function = self.loss_function.to(self.device)
            if self.args.use_gan:
                self.discriminator = nn.DataParallel(self.discriminator)
                self.discriminator = self.discriminator.to(self.device)
                self.gan_criterion = nn.DataParallel(self.gan_criterion)
                self.gan_criterion = self.gan_criterion.to(self.device)

                if 0.0 < self.args.w_gen_seg3d or self.args.use_seg3d_proxy:
                    self.seg_criterion = nn.DataParallel(self.seg_criterion)
                    self.seg_criterion = self.seg_criterion.to(self.device)

        for epoch in range(self.args.epochs):
            if had_training_error:
                print('exiting early due to training error')
                break

            self.adjust_learning_rate(self.optimizer, epoch,
                                      self.args.learning_rate)
            if self.args.use_gan:
                self.adjust_learning_rate(self.disc_optimizer, epoch,
                                          self.args.disc_learning_rate)

            start_time = time.time()
            loss_sum = 0.0
            disc_loss_sum = 0.0

            for (i, data) in enumerate(self.train_loader, 0):
                self.model.train()
                if self.args.use_gan and 0.0 < self.args.w_disc_gan_label:
                    self.discriminator.train()

                self.batch_num += 1

                if ((0) != (self.args.test_interval)) and ((0) == ((self.batch_num) % (self.args.test_interval))):
                    self.run_test_batch(use_file_tuples=False)

                if (not self.args.use_variable_num_views) or (self.batch_num % self.args.log_interval == 0):
                    num_inputs_to_use = self.args.num_combine_views
                else:
                    sample_prob = np.random.uniform(0, 1, 1)[0]
                    if sample_prob < 0.500:
                        num_inputs_to_use = 1
                    elif sample_prob < 0.750:
                        num_inputs_to_use = 2
                    elif sample_prob < 0.875:
                        num_inputs_to_use = 3
                    else:
                        num_inputs_to_use = 4

                data = self.get_data(data, num_inputs_to_use)

                if 0 == (i + 1) % self.args.log_interval:
                    print('start ' + str(i + 1) + ' of ' + str(self.n_img / self.args.batch_size))

                model_out = self.model(num_inputs_to_use, data)

                if self.args.use_gan and 0.0 < self.args.w_disc_gan_label:
                    # reset training params
                    self.disc_optimizer.zero_grad()

                    disc_loss = self.compute_disc_losses(model_out, data, loss_type='train')

                    # update model
                    if self.args.use_amp:
                        with amp.scale_loss(disc_loss, self.disc_optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        disc_loss.backward()
                    self.disc_optimizer.step()

                # reset training params
                self.optimizer.zero_grad()

                loss = self.compute_gen_losses(model_out, data, loss_type='train')

                # update model
                if self.args.use_amp:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                self.optimizer.step()

                loss_item = loss.item()
                if self.args.use_gan and 0.0 < self.args.w_disc_gan_label:
                    disc_loss_item = disc_loss.item()
                else:
                    disc_loss_item = 0.0

                # check for NAN during training
                if loss_item != loss_item or disc_loss_item != disc_loss_item:
                    print('NAN loss in training:', loss_item, disc_loss_item)
                    had_training_error = True
                    exit(-1)

                loss_sum += loss_item

                if self.batch_num % self.args.log_interval == 0:
                    log_string = "Batch %d" % self.batch_num
                    for k, v in self.logs.items():
                        if 'l_eval_' == k[0:7]:
                            scale_factor = 1.0
                        elif 'l_test_' == k[0:7]:
                            scale_factor = (float(self.args.log_interval) / self.args.test_interval) if 0 != self.args.test_interval else 1.0
                        else:
                            scale_factor = float(self.args.log_interval)
                        log_string += " [%s] %5.3f" % (k, v / scale_factor)

                    log_string += ". Took %5.2f" % (time.time() - start_time)

                    print(log_string)

                    for tag, value in self.logs.items():
                        if 'l_eval_' == tag[0:7]:
                            scale_factor = 1.0
                        elif 'l_test_' == tag[0:7]:
                            scale_factor = (float(self.args.log_interval) / self.args.test_interval) if 0 != self.args.test_interval else 1.0
                        else:
                            scale_factor = float(self.args.log_interval)
                        self.logger.scalar_summary(tag, value / scale_factor, self.batch_num)

                    self.reset_logs('train')
                    self.reset_logs('test')
                    self.log_images(model_out, data, 'T_Images')

                if 0 == (i + 1) % self.args.log_interval:
                    crnt_time = time.time()
                    print('end ' + str(i + 1) + ' of ' + str(self.n_img / self.args.batch_size))
                    print(
                        'time:',
                        round(crnt_time - start_time, 3),
                        's',
                        loss_item,
                        loss_sum / self.args.log_interval,
                    )
                    start_time = crnt_time
                    loss_sum = 0.0

                # save regularly after processing the specified number of input images
                if 0 == self.batch_num % self.args.int_save_interval:
                    model_name = self.args.model_path[:-4] + '_int_cpt.pth'
                    self.save(model_name)

                if 0 == self.batch_num % self.args.checkpoint_save_interval:
                    model_name = self.args.model_path[:-4] + '_batch_' + str(int(self.batch_num / 1000.0)) + 'k_cpt.pth'
                    self.save(model_name)
                    print('Checkpoint model saved to ' + model_name)

            if 0 == (epoch + 1) % self.args.epoch_save_interval:
                model_name = self.args.model_path[:-4] + '_epoch_' + str(epoch) + '_cpt.pth'
                self.save(model_name)
                print('Epoch model saved to ' + model_name)

            if self.do_run_eval:
                epoch_test_loss, epoch_raw_ssim_loss = self.run_eval()
                print('Epoch testing loss: ' + str(epoch) + ' ' + str(epoch_test_loss) + ' ' + str(epoch_raw_ssim_loss))
                if epoch_raw_ssim_loss > self.best_epoch_raw_ssim_loss:
                    print('Best raw ssim: ' + str(1.0 - epoch_raw_ssim_loss) + ' ' + str(epoch_raw_ssim_loss))
                    self.best_epoch_raw_ssim_loss = epoch_raw_ssim_loss
                    model_name = self.args.model_path[:-4] + '_best.pth'
                    self.save(model_name)

        print('Finished Training. Best loss: ', self.best_epoch_raw_ssim_loss)
        self.save(self.args.model_path)

    def run_test_batch(self, use_file_tuples=True):
        self.test_batch_num += 1

        self.model.eval()

        if self.args.use_gan:
            self.discriminator.eval()

        with torch.no_grad():
            if use_file_tuples:
                targetLabel = "Static_Images"
                # reset iterator
                self.file_test_loader_iter = iter(self.file_test_loader)
                test_data = next(self.file_test_loader_iter)
            else:
                targetLabel = "Images"
                try:
                    test_data = next(self.test_loader_iter)
                except:
                    # reset iterator
                    self.test_loader_iter = iter(self.test_loader)
                    test_data = next(self.test_loader_iter)

            num_inputs_to_use = self.args.num_combine_views

            data = self.get_data(test_data, num_inputs_to_use)

            model_out = self.model(num_inputs_to_use, data)
            test_loss = self.compute_gen_losses(model_out, data, loss_type='test')

            self.log_images(model_out, data, targetLabel)
            return test_loss

    def log_images(self, model_out, data, targetLabel='T_Images'):
        src_rgb_image = data['src_rgb_image']
        tgt_rgb_image = data['tgt_rgb_image'][0]

        if self.args.use_seg3d_proxy:
            src_seg_image = data['src_seg_image']
            tgt_seg_image = data['tgt_seg_image'][0]

            gen_tgt_seg3d = model_out[2][0]
            gen_src_seg3d = model_out[1]
            src_cat_images = torch.cat((src_rgb_image[0],
                                        torch.cat((gen_src_seg3d[0], gen_src_seg3d[0], gen_src_seg3d[0]), 1),
                                        torch.cat((src_seg_image[0], src_seg_image[0], src_seg_image[0]), 1)), 3)
            for view_idx in range(1, self.args.num_combine_views):
                src_cat_images = torch.cat((src_cat_images, src_rgb_image[view_idx],
                                            torch.cat((gen_src_seg3d[view_idx], gen_src_seg3d[view_idx],
                                                       gen_src_seg3d[view_idx]), 1),
                                            torch.cat((src_seg_image[view_idx], src_seg_image[view_idx],
                                                       src_seg_image[view_idx]), 1)), 3)

            tgt_seg_rgb = torch.cat((tgt_seg_image, tgt_seg_image, tgt_seg_image), 1)
            gen_tgt_seg3d_rgb = torch.cat((gen_tgt_seg3d, gen_tgt_seg3d, gen_tgt_seg3d), 1)
            cat_images = torch.cat((src_cat_images[:, 0:3, :, :],
                                    model_out[0][:, 0:3, :, :], tgt_rgb_image[:, 0:3, :, :],
                                    gen_tgt_seg3d_rgb[:, 0:3, :, :],
                                    tgt_seg_rgb[:, 0:3, :, :]), 3)
        else:
            src_cat_images = src_rgb_image[0]
            for view_idx in range(1, self.args.num_combine_views):
                src_cat_images = torch.cat((src_cat_images, src_rgb_image[view_idx]), 3)

            cat_images = torch.cat((src_cat_images[:, 0:3, :, :],
                                    model_out[0][:, 0:3, :, :], tgt_rgb_image[:, 0:3, :, :]), 3)

        self.logger.image_summary(targetLabel, self.images_to_numpy(cat_images), self.batch_num)

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
                    if self.args.use_seg3d_proxy:
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
                    src_seg_image = data['src_seg_image']

                    tgt_rgb_image = data['tgt_rgb_image'][0]
                    tgt_seg_image = data['tgt_seg_image'][0]

                    src_cat_images = torch.cat((src_rgb_image[0], torch.cat((src_seg_image[0], src_seg_image[0], src_seg_image[0]), 1)),
                                               3)
                    for view_idx in range(1, num_inputs_to_use):
                        src_cat_images = torch.cat((src_cat_images, src_rgb_image[view_idx],
                                                    torch.cat((src_seg_image[view_idx], src_seg_image[view_idx], src_seg_image[view_idx]),
                                                              1)), 3)

                    tgt_seg_rgb = torch.cat((tgt_seg_image, tgt_seg_image, tgt_seg_image), 1)
                    gen_tgt_seg3d = model_out[2][0]
                    gen_tgt_seg3d_rgb = torch.cat((gen_tgt_seg3d, gen_tgt_seg3d, gen_tgt_seg3d), 1)
                    cat_images = torch.cat((src_cat_images[:, 0:3, :, :],
                                            model_out[0][:, 0:3, :, :], tgt_rgb_image[:, 0:3, :, :],
                                            gen_tgt_seg3d_rgb[:, 0:3, :, :], tgt_seg_rgb[:, 0:3, :, :]), 3)

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
        return {'l_train_disc': 0.0,
                'l_train_disc_gan': 0.0,
                'l_train_disc_running': 0.0,
                'l_train_gen': 0.0,
                'l_train_gen_gan': 0.0,
                'l_train_gen_l1': 0.0,
                'l_train_gen_raw_l1': 0.0,
                'l_train_gen_raw_ssim': 0.0,
                'l_train_gen_ssim': 0.0,
                'l_train_gen_seg3d': 0.0,
                'l_train_gen_vgg': 0.0,
                'l_train_gen_running': 0.0,
                'l_test_gen': 0.0,
                'l_test_gen_gan': 0.0,
                'l_test_gen_l1': 0.0,
                'l_test_gen_ssim': 0.0,
                'l_test_gen_raw_l1': 0.0,
                'l_test_gen_raw_ssim': 0.0,
                'l_test_gen_seg3d': 0.0,
                'l_test_gen_vgg': 0.0,
                'l_test_gen_running': 0.0,
                'l_eval_gen': 0.0,
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
        if not self.args.use_amp:
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

    def compute_disc_losses(self, model_out, data, loss_type='train'):
        tgt_rgb_image = data['tgt_rgb_image'][0]

        fake_labels, _ = self.discriminator(model_out[0])
        real_labels, _ = self.discriminator(tgt_rgb_image)
        self.ones = self.ones_like(fake_labels, device=self.device)
        self.zeros = self.zeros_like(fake_labels, device=self.device, val=self.fake_val)

        l_disc_gan = self.args.w_disc_gan_label * \
                     (self.gan_criterion(real_labels, self.ones) + \
                      self.gan_criterion(fake_labels, self.zeros)).mean()

        l_disc_gan_item = l_disc_gan.item()

        self.logs['l_' + loss_type + '_disc_gan'] += l_disc_gan_item
        self.logs['l_' + loss_type + '_disc'] += l_disc_gan_item

        if 'train' == loss_type:
            self.total_disc_loss_sum += l_disc_gan_item
            running_disc_loss = self.total_disc_loss_sum / self.batch_num
            self.logs['l_' + loss_type + '_disc_running'] += running_disc_loss

        return l_disc_gan

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

    def save(self, path, save_optimizer=True):
        try:
            disc_path = path[:-4] + '_disc.pth'

            print('saving ' + str(path) + '...')
            if isinstance(self.model, torch.nn.DataParallel):
                torch.save(self.model.module, path)
            else:
                torch.save(self.model, path)

            if self.args.use_gan:
                if isinstance(self.discriminator, torch.nn.DataParallel):
                    torch.save(self.discriminator.module, disc_path)
                else:
                    torch.save(self.discriminator, disc_path)

            if save_optimizer:
                opt_path = path[:-4] + '_opt.pth'
                torch.save(self.optimizer, opt_path)

                if self.args.use_gan:
                    disc_opt_path = path[:-4] + '_disc_opt.pth'
                    torch.save(self.disc_optimizer, disc_opt_path)
        except:
            print('FAILED saving ' + str(path) + ' with new error, continuing...')
            return

    def load(self, path, load_disc=True, in_disc_path='', load_optimizer=False):
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

        if load_optimizer:
            print('loading optimizer from saved checkpoint...')

            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                        betas=self.adam_betas, eps=self.adam_eps)
            opt_path = path[:-4] + '_opt.pth'
            self.optimizer.load_state_dict(torch.load(opt_path).state_dict())

            if self.args.use_gan and load_disc:
                self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.args.disc_learning_rate,
                                                 betas=self.adam_betas, eps=self.adam_eps)
                disc_opt_path = path[:-4] + '_disc_opt.pth'
                self.disc_optimizer.load_state_dict(torch.load(disc_opt_path).state_dict())
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate,
                                        betas=self.adam_betas, eps=self.adam_eps)
            if self.args.use_gan and load_disc:
                self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=args.disc_learning_rate,
                                                 betas=self.adam_betas, eps=self.adam_eps)
