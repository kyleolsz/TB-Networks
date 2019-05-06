import torch
from tester import TBNTester
from init_args import ArgLoader


def main():
    in_config_list = 'test_config.ini'
    args = ArgLoader.return_arguments(in_config_list)

    model = TBNTester(args)
    model.load(args.input_model_file, load_disc=args.use_gan)
    model.run_eval(args.num_combine_views)


if __name__ == '__main__':
    main()
