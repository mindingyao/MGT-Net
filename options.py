import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--n_color', type=int, default=3)
parser.add_argument('--num_classes', type=int, default=1)
parser.add_argument('--img_size', type=int, default=28, help='input patch size of network input')
parser.add_argument('--vit_name', type=str, default='ViT-B_16')
parser.add_argument('--vit_patches_size', type=int, default=11, help='vit_patches_size, default is 16')

config = parser.parse_args()
