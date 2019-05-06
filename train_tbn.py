import os
import torch
from trainer import TBNTrainer
from init_args import ArgLoader


def main():
    in_config_list = 'train_config.ini'
    args = ArgLoader.return_arguments(in_config_list)

    if args.test_interval > args.log_interval:
        args.test_interval = args.log_interval

    if False == args.use_seg3d_proxy:
        args.w_gen_seg3d = 0.0

    model = TBNTrainer(args)
    load_disc = (args.use_gan and args.load_discriminator)
    if args.load_model and '' != args.input_model_file:
        model.load(args.input_model_file, load_disc=load_disc, load_optimizer=args.load_optimizer)

    # params to check if we should restart existing task
    if args.continue_train:
        # see if checkpoint from previous run exists
        model_name = args.model_path[:-4] + '_int_cpt.pth'
        if os.path.exists(model_name):
            print('continuing training, loading model and optimizer from: ' + model_name)
            # continuing run, load models and optimizer
            model.load(model_name, load_disc=load_disc, load_optimizer=args.load_optimizer)
    model.train()


if __name__ == '__main__':
    main()
