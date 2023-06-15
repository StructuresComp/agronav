import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes

from segmentation.agronav_mobilenetv3 import *
# from segmentation.agronav_hrnet import *
# from segmentation.agronav_resnest import *


import glob as glob
import os
import pudb
import mmcv
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--image', 
        default='segmentation/data/demo/GOPR0016.JPG',
        help='Image file')
    parser.add_argument(
        '-w', '--checkpoint', 
        default='segmentation/checkpoints/MobileNetV3.pth',
        help='weight file name'
    )
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--model',
        default='MobileNetV3',
        help='Model name')
    args = parser.parse_args()

    # if args.model == 'ResNest':
    #     cfg = cfg_resnest
    # elif args.model == 'HRNet':
    #     cfg = cfg_hrnet
    # else:
    #     cfg = cfg_mobilenetv3

    # build the model from a config file and a checkpoint file
    model = init_segmentor(cfg, checkpoint=None, device=args.device)
    cfg.load_from = args.checkpoint
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)

    image_path = args.image
    save_dir = 'segmentation/demo'
    image = mmcv.imread(image_path)
    result= inference_segmentor(model, image)

    save_name = f"{image_path.split(os.path.sep)[-1].split('.')[0]}"

    show_result_pyplot( model,
                        image,
                        result,
                        palette,
                        opacity=args.opacity,
                        out_file=f"{save_dir}/{save_name}.jpg")


if __name__ == '__main__':
    main()