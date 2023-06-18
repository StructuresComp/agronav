import argparse
import os
import random
import time
from os.path import isfile, join, split

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import numpy as np
import tqdm
import yaml
import cv2

from torch.optim import lr_scheduler
from logger import Logger
from dataloader import get_loader
from model.network import Net
from skimage.measure import label, regionprops
from utils import reverse_mapping, visulize_mapping, edge_align, get_boundary_point

parser = argparse.ArgumentParser(description='PyTorch Semantic-Line Training')
# arguments from command line
parser.add_argument('--config', default="./config.yml", help="path to config file")
parser.add_argument('--model', required=True, help='path to the pretrained model')
parser.add_argument('--align', default=False, action='store_true')
parser.add_argument('--tmp', default="", help='tmp')
args = parser.parse_args()

assert os.path.isfile(args.config)
CONFIGS = yaml.full_load(open(args.config))

# merge configs
if args.tmp != "" and args.tmp != CONFIGS["MISC"]["TMP"]:
    CONFIGS["MISC"]["TMP"] = args.tmp

os.makedirs(CONFIGS["MISC"]["TMP"], exist_ok=True)
logger = Logger(os.path.join(CONFIGS["MISC"]["TMP"], "log.txt"))


def main():
    logger.info(args)

    seg_model = init_segmentor(CONFIGS["SEGMENTATION"]["CONFIG_PATH"], checkpoint=None,
                               device=CONFIGS["TRAIN"]["DEVICE"])
    load_checkpoint(seg_model, CONFIGS["SEGMENTATION"]["CHECKPOINT"], map_location='cpu')

    second_model = Net(numAngle=CONFIGS["SECOND_MODEL"]["NUMANGLE"], numRho=CONFIGS["SECOND_MODEL"]["NUMRHO"],
                       backbone=CONFIGS["SECOND_MODEL"]["BACKBONE"])
    second_model.load_state_dict(torch.load(CONFIGS["SECOND_MODEL"]["WEIGHTS"]))
    second_model.to(CONFIGS["TRAIN"]["DEVICE"])
    second_model.eval()

    test_loader = get_loader(CONFIGS["DATA"]["TEST_DIR"], CONFIGS["DATA"]["TEST_LABEL_FILE"],
                             batch_size=1, num_thread=CONFIGS["DATA"]["WORKERS"], test=True)

    logger.info("Data loading done.")

    logger.info("Start testing.")
    total_time = test(test_loader, seg_model, second_model, args)

    logger.info("Test done! Total %d images at %.4f seconds without image I/O, FPS: %.3f" % (
    len(test_loader), total_time, len(test_loader) / total_time))


def test(test_loader, seg_model, second_model, args):
    seg_model.eval()
    second_model.eval()
    with torch.no_grad():
        bar = tqdm.tqdm(test_loader)
        forward_time = 0
        post_processing_time = 0
        for i, data in enumerate(bar):
            t = time.time()
            images, names, size = data

            images = images.to(CONFIGS["TRAIN"]["DEVICE"])
            segmentation_map = perform_segmentation(seg_model, images)

            # Perform inference on the segmentation map using the second model
            output = perform_inference(second_model, segmentation_map)

            forward_time += (time.time() - t)
            t = time.time()
            visualize_save_path = os.path.join(CONFIGS["MISC"]["TMP"], 'visualize_test')
            os.makedirs(visualize_save_path, exist_ok=True)

            # Save the output image
            save_output(output, join(visualize_save_path, names[0].split('/')[-1]))

            post_processing_time += (time.time() - t)
    print('Forward time for total images: %.6f' % forward_time)
    print('Post-processing time for total images: %.6f' % post_processing_time)
    return forward_time + post_processing_time


if __name__ == '__main__':
    main()

