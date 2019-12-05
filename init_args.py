import configargparse


def parse_boolean(b):
    if len(b) < 1:
        raise ValueError('Cannot parse empty string into boolean.')
    b = b[0].lower()
    if b == 't' or b == 'y' or b == '1':
        return True
    if b == 'f' or b == 'n' or b == '0':
        return False
    raise ValueError('Cannot parse string into boolean.')


class ArgLoader(object):

    @staticmethod
    def return_arguments(in_config_list='config.ini'):
        parser = configargparse.ArgParser(default_config_files=[in_config_list],
                                          description='PyTorch Transformable Bottleneck Network.')

        parser.add_argument('--model_path', type=str, default='output/Models/output_model.pth',
                            help='Path for output model file.')
        parser.add_argument('--input_height', type=int, default=160,
                            help='Height of images used as input to network.')
        parser.add_argument('--input_width', type=int, default=160,
                            help='Width of images used as input to network.')
        parser.add_argument('--epochs', type=int, default=3000,
                            help='Number of epochs for training.')
        parser.add_argument('--learning_rate', type=float, default=2e-4,
                            help='Learning rate for generator network.')
        parser.add_argument('--disc_learning_rate', type=float, default=2e-4,
                            help='Learning rate for discriminator network.')
        parser.add_argument('--batch_size', type=int, default=8,
                            help='Number of images to use per batch.')
        parser.add_argument('--num_workers', type=int, default=1,
                            help='Number of worker threads to use for loading images.')
        parser.add_argument('--device_mode', type=str, default='gpu',
                            help='Device to use (gpu or cpu).')
        parser.add_argument('--cuda_device_num', type=int, default=0,
                            help='Device ID of GPU to use. Use 0 when training with with --use_data_parallel enabled for multi-gpu training.')
        parser.add_argument('--print_output', type=parse_boolean, default=False,
                            help='Enables printing of output images and/or occupancy volumes during evaluation.')
        parser.add_argument('--print_seg_output', type=parse_boolean, default=False,
                            help='Enables printing of segmentation masks when --print_output is enabled.')
        parser.add_argument('--print_occupancy_volume', type=parse_boolean, default=False,
                            help='Enables printing of occupancy volumes during evaluation when --print_output is enabled.')
        parser.add_argument('--img_out_dir', type=str, default='imgs.out/',
                            help='Directory in which to write output images when --print_output is enabled')
        parser.add_argument('--log_folder', type=str, default='logs/tboard/output_log/',
                            help='Directory in which tensorboard logs are written.')
        parser.add_argument('--log_interval', type=int, default=100,
                            help='Batch interval at which to print loss statistics.')
        parser.add_argument('--test_interval', type=int, default=100,
                            help='Batch interval at which to run image from test set.')
        parser.add_argument('--epoch_test_interval', type=int, default=1,
                            help='Epoch interval at which to run evaluation.')
        parser.add_argument('--vgg_model_path', default='',
                            help='Path to VGG model. If blank, model will be downloaded automatically.')
        parser.add_argument('--w_gen_l1', type=float, default=1.0,
                            help='Weight of L1 loss.')
        parser.add_argument('--w_gen_vgg', type=float, default=5.0,
                            help='Weight of MSE loss on VGG features.')
        parser.add_argument('--w_gen_ssim', type=float, default=10.0,
                            help='Weight of Structural Similarity Index (SSIM) loss.')
        parser.add_argument('--w_gen_gan_label', type=float, default=0.05,
                            help='Weight of adversarial loss in generator.')
        parser.add_argument('--w_gen_seg3d', type=float, default=10.0,
                            help='Weight of segmentation loss.')
        parser.add_argument('--w_disc_gan_label', type=float, default=1.0,
                            help='Weight of adversarial loss in discriminator.')
        parser.add_argument('--load_model', type=parse_boolean, default=False,
                            help='If true, model specified by --input_model_file will be loaded at initialization.')
        parser.add_argument('--continue_train', type=parse_boolean, default=True,
                            help='If enabled, will load latest checkpoint of the model specified by --model_path.')
        parser.add_argument('--input_model_file', type=str, default='input/Models/input_model.pth',
                            help='Model to load.')
        parser.add_argument('--use_gan', type=parse_boolean, default=True,
                            help='Enables use of adversarial loss.')
        parser.add_argument('--use_amp', type=parse_boolean, default=False,
                            help='Enables use of NVIDIA\'s Apex for automatic mixed precision. Must have Apex installed: https://github.com/NVIDIA/apex.')
        parser.add_argument('--normalize_ssim_loss', type=parse_boolean, default=True,
                            help='Normalizes SSIM loss to range [0.0, 2.0].')
        parser.add_argument('--use_gan_noise', type=parse_boolean, default=False,
                            help='Enables use of noise in discriminator.')
        parser.add_argument('--gan_noise_sigma', type=float, default=0.2,
                            help='Sigma value for noise in discriminator when --use_gan_noise is enabled.')
        parser.add_argument('--num_input_channels', type=int, default=3,
                            help='Number of input channels for generator. 3 for RGB, 4 for RGBA.')
        parser.add_argument('--num_output_channels', type=int, default=4,
                            help='Number of output channels for generator. 3 for RGB, 4 for RGBA.')
        parser.add_argument('--num_features', type=int, default=800,
                            help='Number of 2D feature maps created by the 2D encoder. These maps will be reshaped to produce a 3D volume with a vector of size (num_features/vol_dim) in each cell.')
        parser.add_argument('--vol_dim', type=int, default=0,
                            help='Resolution of volumetric bottleneck. 0 means infer volume dimension from other params.')
        parser.add_argument('--print_args', type=parse_boolean, default=True,
                            help='Whether to print list of arguments at initialization.')
        parser.add_argument('--int_save_interval', type=int, default=1000,
                            help='Interval at which to save a snapshot of the latest model. Will be overwritten at the specified number of iterations. Set to 0 to disable.')
        parser.add_argument('--checkpoint_save_interval', type=int, default=30000,
                            help='A checkpoint will be saved after the specified number of iterations. Set to 0 to disable.')
        parser.add_argument('--epoch_save_interval', type=int, default=1,
                            help='A checkpoint will be saved after the specified number of epochs. Set to 0 to disable.')
        parser.add_argument('--num_combine_views', type=int, default=4,
                            help='Number of input views to use.')
        parser.add_argument('--dataset_name', type=str, default='chair',
                            help='Dataset to use. Options are: chair, car, drc_chair, and drc_car.')
        parser.add_argument('--img_path', type=str, default='datasets/shapenet',
                            help='Path to directory containing input images.')
        parser.add_argument('--use_data_parallel', type=parse_boolean, default=True,
                            help='Enables use of multi-GPU training.')
        parser.add_argument('--cull_identity_transform', type=parse_boolean, default=True,
                            help='Determines whether identity transforms are included during training.')
        parser.add_argument('--use_ls_gan', type=parse_boolean, default=False,
                            help='Enables use of LS-GAN loss when --use_gan is enabled.')
        parser.add_argument('--num_input_convs', type=int, default=1,
                            help='Number of additional input convolutions to apply. Each additional convolution reduces the input size by a factor of 2.')
        parser.add_argument('--num_output_deconvs', type=int, default=1,
                            help='Number of additional output convolutions to apply. Each additional convolution increases the output size by a factor of 2.')
        parser.add_argument('--upsample_output', type=parse_boolean, default=True,
                            help='Whether to upsample output to final resolution specified by --final_width and --final_height before measuring loss.')
        parser.add_argument('--final_height', type=int, default=256,
                            help='Height to which output images will be upsampled when --upsample_output is enabled.')
        parser.add_argument('--final_width', type=int, default=256,
                            help='Width to which output images will be upsampled when --upsample_output is enabled.')
        parser.add_argument('--use_variable_num_views', type=parse_boolean, default=False,
                            help='Whether to randomly choose the number of views to use during training. If disabled, --num_combine_views will be used at each iteration.')
        parser.add_argument('--gan_num_extra_layers', type=int, default=2,
                            help='Number of additional convolution layers to add to the discriminator. Additional layers reduce the total number of discrimantor output values.')
        parser.add_argument('--use_seg3d_proxy', type=parse_boolean, default=True,
                            help='Enables use of segmentation loss compared to the input image foreground masks.')
        parser.add_argument('--use_seg3d_softmax', type=parse_boolean, default=True,
                            help='Use softmax operation on output of 3D segmentation branch.')
        parser.add_argument('--use_synthetic_input', type=parse_boolean, default=False,
                            help='When enabled, one real input image will be used with additional input images synthesized by the network.')
        parser.add_argument('--use_random_transforms', type=parse_boolean, default=False,
                            help='Enables use of random poses when --use_synthetic_input is enabled. When false, regularly sampled poses around the vertical axis will be used.')
        parser.add_argument('--azim_rotation_angle_increment', type=float, default=10.0,
                            help='Increments between azimuth angles sampled during training/testing.')
        parser.add_argument('--elev_rotation_angle_increment', type=float, default=10.0,
                            help='Increments between elevation angles sampled during training/testing.')
        parser.add_argument('--crop_x_dim', type=int, default=0,
                            help='Offset by which to crop input images in the x-dimension.')
        parser.add_argument('--crop_y_dim', type=int, default=0,
                            help='Offset by which to crop input images in the y-dimension.')
        parser.add_argument('--num_res_convs', type=int, default=2,
                            help='Number of residual convolution layers to use in residual blocks.')
        parser.add_argument('--use_src_bg_noise', type=parse_boolean, default=False,
                            help='Add background noise to source images.')
        parser.add_argument('--use_tgt_bg_noise', type=parse_boolean, default=False,
                            help='Add background noise to target images.')
        parser.add_argument('--load_discriminator', type=parse_boolean, default=True,
                            help='Load the saved discriminator model when loading a saved model.')
        parser.add_argument('--load_optimizer', type=parse_boolean, default=True,
                            help='Load the saved optimizer when loading a saved model.')
        parser.add_argument('--num_gen_features', type=int, default=32,
                            help='Base number of generator features to use.')
        parser.add_argument('--encode_feature_scale_factor', type=int, default=1,
                            help='Factor by which to reduce number of features used in the encoder.')
        parser.add_argument('--decode_feature_scale_factor', type=int, default=1,
                            help='Factor by which to reduce number of features used in the decoder.')
        parser.add_argument('--use_elev_transform', type=parse_boolean, default=True,
                            help='Enable elevation variation in input and output images.')
        parser.add_argument('--elev_transform_threshold', type=float, default=0.5,
                            help='Probability of using variable elevations in input and output images.')

        args = parser.parse_args()

        return args
