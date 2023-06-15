import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
from agronav_test import *

import glob as glob
import os
import pudb
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    parser.add_argument(
        '-i', '--input', default='data/demo/',
        help='path to the input data'
    )
    parser.add_argument(
        '-w', '--weights', 
        default='checkpoints/MobileNetV3.pth',
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
    args = parser.parse_args()

    # cfg = Config.fromfile('configs/mobilenet_v3/lraspp_m-v3-d8_512x1024_320k_cityscapes.py')
    # build the model from a config file and a checkpoint file
    model = init_segmentor(cfg, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.weights, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)

    # palette = get_palette(args.palette)
    # pu.db

    image_paths = glob.glob(f"{args.input}/*.JPG")
    save_dir = 'demo'

    for image_path in tqdm(image_paths):
    # for i, image_path in enumerate(image_paths):
        image = mmcv.imread(image_path)
        result= inference_segmentor(model, image)
        # pu.db

        save_name = f"{image_path.split(os.path.sep)[-1].split('.')[0]}"

        show_result_pyplot( model,
                            image,
                            result,
                            palette,
                            opacity=args.opacity,
                            out_file=f"{save_dir}/{save_name}.jpg")


if __name__ == '__main__':
    main()