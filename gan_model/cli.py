import os
from argparse import ArgumentParser
from math import log2
from typing import Union

from pl_fun import generate, to_onnx, train
from utils import check_aspect_ratio


def check_image_size(width: int, height: Union[int, None] = None) -> str:
    if height is None:
        height = width

    aspect_ratio = check_aspect_ratio(width, height)
    if aspect_ratio not in ['1:1', '4:3']:
        return ('Image aspect ratio is not 1:1 or 4:3')
    elif aspect_ratio == '1:1':
        if not log2(width).is_integer():
            return ('Image width is not power of 2')
    elif aspect_ratio == '4:3':
        if width % 20 != 0:
            return ('Image width is not multiply of 20')
        if width < 20:
            return ('Image width less than 20 pixels')
        if not log2(width//20).is_integer():
            return ('Image width is not power of 2')

    return ''


def check_args(args):
    print('Checking argument...')
    total_violation = 0
    int_args = [
        args.n_z,
        args.n_gf,
        args.n_df,
        args.batch_size,
        args.n_epochs,
        args.save_freq,
        args.n_critics,
    ]
    float_args = [
        args.aug_prob,
        args.g_lr,
        args.d_lr,
    ]

    # TRAIN ARGS
    if args.train:
        if not os.path.exists(args.image_path):
            print(f'Directory path {args.image_path} is not exist')
            total_violation += 1

        if len(args.im_size) > 2:
            print(f'Length of image size should be 1 or 2')
            total_violation += 1
        else:
            message = check_image_size(*args.im_size)
            if message != '':
                print(message)
                total_violation += 1

        if args.train and args.loss_type == 1 and args.lambda_gp <= 0.0:
            print(f'Lambda GP should be >= 0.0')
            total_violation += 1

        for arg_value in int_args:
            if arg_value <= 0:
                arg_name = f'{arg_value=}'.partition('=')[0]
                print(f'Value of argument {arg_name} should be > 0')
                total_violation += 1

        for arg_value in float_args:
            if arg_value < 0.0 or arg_value > 1.0:
                arg_name = f'{arg_value=}'.partition('=')[0]
                print(f'Value of argument {arg_name} should be between 0.0 - 1.0')
                total_violation += 1

    if args.generate:
        if args.total_image <= 0:
            print(f'Total generated image should be > 0')
            total_violation += 1
        if args.grouped_image is not None and args.grouped_image <= 0:
            print(f'Quantity of image to create big image should be > 0')
            total_violation += 1

    if total_violation > 0:
        print(f'{total_violation} on argument detected, exiting...')
        exit()


if __name__ == '__main__':
    parser = ArgumentParser(description='Train DCGAN')

    # MODE
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', help='Train GAN', action='store_true')
    group.add_argument('--generate', help='Generate image',
                       action='store_true')
    group.add_argument(
        '--to_onnx', help='Convert Generator to ONNX', action='store_true')

    # BASE PARSER
    parser.add_argument('--name', help='Name of GAN model',
                        type=str, default='OMORI_GAN')
    parser.add_argument('--accelerator', type=str,
                        default='gpu', choices=('cpu', 'gpu'))
    parser.add_argument(
        '--ckpt_name', help='Specify checkpoint name used. If not specified, use latest one.', type=str, default=None)

    # TRAIN
    p_input = parser.add_argument_group('Dataset')
    p_input.add_argument('--image_path', help='Path to image directory', metavar='directory',
                         type=str, default='../_dataset/processed_img')
    p_input.add_argument('--im_size', help='Image size', metavar=('W', 'H'),
                         type=int, default=[640, 480], nargs='+')
    p_input.add_argument(
        '--transparent', help='Train GAN to generate transparent image', action='store_true')

    p_aug = parser.add_argument_group('Augmentation')
    p_aug.add_argument('--aug_level', help='0: No augmentation, 1: Only real image for discriminator, 2: DiffAug',
                       type=int, default=2, choices=(0, 1, 2))
    p_aug.add_argument('--aug_prob', help='Augmentation probability (only for level 1)',
                       type=float, default=0.3, metavar='float')
    p_aug.add_argument('--aug_type', help='Type of augmentation used',
                       type=str, nargs='*', default=('color', 'translation', 'cutout' 'hflip'),
                       choices=('color', 'translation', 'cutout', 'hflip'))

    p_model = parser.add_argument_group('Model architecture')
    p_model.add_argument('--n_z', help='Dimension of latent vector/input shape of generator',
                         type=int, default=32, metavar='int')
    p_model.add_argument('--n_gf', help='Feature maps for generator',
                         type=int, default=64, metavar='int')
    p_model.add_argument('--n_df', help='Feature maps for discriminator',
                         type=int, default=64, metavar='int')
    p_model.add_argument('--g_conv_type', help='0: Transpose conv, 1: Upscale and conv.',
                         type=int, default=0, choices=(0, 1))
    p_model.add_argument('--g_upscale_type', help='0: Nearest, 1: Bilinear.',
                         type=int, default=0, choices=(0, 1))
    p_model.add_argument('--d_norm_type', help='0: Instance norm, 1: Spectral norm',
                         type=int, default=0, choices=(0, 1))

    p_train = parser.add_argument_group('Model training')
    p_train.add_argument(
        '--use_aim', help='Use Aim to track model training', action='store_true')
    p_train.add_argument('--batch_size', type=int, default=64, metavar='int')
    p_train.add_argument('--n_epochs', help='Total training epoch',
                         type=int, default=10000, metavar='int')
    p_train.add_argument('--loss_type', help='0: BCE, 1: WGAN-GP',
                         type=int, default=0, choices=(0, 1))
    p_train.add_argument('--smoothing_value', help='value of label smoothing',
                         type=float, default=0.05)
    p_train.add_argument('--z_distibution', help='Distribution type of latent vector. 0: -1.0 to 1.0, 1: 0.0 to 1.0',
                         type=int, default=0, choices=(0, 1))
    p_train.add_argument('--save_freq', help='Epoch frequency to save model and show progress',
                         type=int, default=100, metavar='int')
    p_train.add_argument('--g_lr', help='Generator LR',
                         type=float, default=0.0001, metavar='float')
    p_train.add_argument('--d_lr', help='Discriminator LR',
                         type=float, default=0.0001, metavar='float')
    p_train.add_argument('--prune', help='Prune model',
                         action='store_true')
    p_train.add_argument('--quantize', help='Quantize model',
                         action='store_true')
    # WGAN-GP specific
    p_train.add_argument('--n_critics', help='Total critic of discriminator (WGAN-GP only)',
                         type=int, default=5, metavar='int')
    p_train.add_argument('--lambda_gp', help='Lambda value for Gradient Penalty (WGAN-GP only)',
                         type=float, default=10.0, metavar='float')

    # PREDICT
    p_predict = parser.add_argument_group('Generate image')
    p_predict.add_argument(
        '--total_image', help='Total generated image', type=int, default=128, metavar='int')
    p_predict.add_argument('--output_path', type=str,
                           default='./predict', metavar='directory')

    # PARSE
    args = parser.parse_args()
    print(args)
    check_args(args)

    if args.train:
        print('Start train...')
        train(args)
    elif args.generate:
        print('Start generate...')
        generate(args)
    elif args.to_onnx:
        print(f'Converting model to ONNX format...')
        to_onnx(args)
