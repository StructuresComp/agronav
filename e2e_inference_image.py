import argparse
import os
import random
import time
import importlib
from os.path import isfile, join, split
import glob as glob
import cv2
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import numpy as np
import tqdm
import yaml
import pudb

from skimage.measure import label, regionprops

from lineDetection.dataloader import get_loader
from lineDetection.logger import Logger
from lineDetection.model.network import Net
from lineDetection.utils import reverse_mapping, visulize_mapping, edge_align, get_boundary_point

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from segmentation.agronav_mobilenetv3 import *


def parse_arguments():
    parser = argparse.ArgumentParser(description='Semantic Segmentation and Semantic-Line Detection')
    parser.add_argument('--input_dir', default='./inference/input/', help="Input images diretory")
    parser.add_argument('--segmentation_model', default='./segmentation/checkpoint/MobileNetV3.pth',
                        help='Semantic segmentation model file')
    parser.add_argument('--segmentation_config', default='./segmentation/agronav_mobilenetv3.py',
                        help='Semantic segmentation config file')
    parser.add_argument('--line_detection_model', default='./lineDetection/checkpoint/model_best.pth',
                        help='Semantic-line detection model file')
    parser.add_argument('--align', default=False, action='store_true', help='Enable alignment')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument('--output_dir', default='./inference/output/', help='Output images directory')

    return parser.parse_args()


args = parse_arguments()

# Semantic Segmentation
# importlib.import_module(parser.get_default('segmentation_config'), package='agronav')
# segmentation_cfg = cfg
# segmentation_cfg = Config.fromfile('segmentation/configs/mobilenet_v3/mobilenet_v3_large_1x_coco.py')
segmentation_model = init_segmentor(segmentation_cfg, checkpoint=None, device='cuda:0')

segmentation_cfg.load_from = args.segmentation_model
checkpoint = load_checkpoint(segmentation_model, args.segmentation_model, map_location='cpu')
if 'CLASSES' in checkpoint.get('meta', {}):
    segmentation_model.CLASSES = checkpoint['meta']['CLASSES']
else:
    segmentation_model.CLASSES = get_classes(args.palette)

# segmentation_checkpoint = load_checkpoint(segmentation_model, args.segmentation_model, map_location='cpu')

# Semantic-Line Detection
line_detection_cfg = yaml.full_load(open('lineDetection/config.yml'))
line_detection_model = Net(numAngle=line_detection_cfg["MODEL"]["NUMANGLE"],
                           numRho=line_detection_cfg["MODEL"]["NUMRHO"],
                           backbone=line_detection_cfg["MODEL"]["BACKBONE"])
line_detection_model = line_detection_model.cuda(device=line_detection_cfg["TRAIN"]["GPU_ID"])
line_detection_checkpoint = torch.load(args.line_detection_model)


def main():
    if 'state_dict' in line_detection_checkpoint.keys():
        line_detection_model.load_state_dict(line_detection_checkpoint['state_dict'])
    else:
        line_detection_model.load_state_dict(line_detection_checkpoint)

    # Data Loader
    test_loader = get_loader(line_detection_cfg["DATA"]["TEST_DIR"], line_detection_cfg["DATA"]["TEST_LABEL_FILE"],
                             batch_size=1, num_thread=line_detection_cfg["DATA"]["WORKERS"], test=True)

    # Semantic Segmentation
    ntime = 0
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".jpg") or filename.endswith(".JPG") or filename.endswith(".png"):
            image = mmcv.imread(args.input_dir + filename)
            width = 800
            image = cv2.resize(image, (width, int((width / image.shape[1]) * image.shape[0])))
            t = time.time()
            segmentation_result = inference_segmentor(segmentation_model, image)
            ntime += (time.time() - t)
            output_file = os.path.join('inference/temp/', os.path.split(filename)[1])
            show_result_pyplot(segmentation_model, image, segmentation_result, palette, show=False, opacity=0.5,
                               out_file=output_file)
            continue
        else:
            continue

    print('Semantic segmentation time for total images: %.6f' % ntime)

    # Extract and save file names
    with open('./inference/inference_filenames.txt', 'w') as f:
        for file in glob.glob(os.path.join('inference/temp', '*.jpg')):
            filename = os.path.split(file)[1]
            filename = os.path.splitext(filename)[0]

            f.write('temp/' + filename + "\n")

    # Semantic-Line Detection
    total_time = run_line_detection(test_loader, line_detection_model, args)

    # Visualize centerline
    save_centerline(args.output_dir, 'inference/output_centerline/')


def run_line_detection(test_loader, model, args):
    # Switch to evaluation mode
    model.eval()
    with torch.no_grad():
        bar = tqdm.tqdm(test_loader)
        iter_num = len(test_loader.dataset)
        ftime = 0
        ntime = 0
        for i, data in enumerate(bar):
            t = time.time()
            images, names, size = data
            images = images.cuda(device=line_detection_cfg["TRAIN"]["GPU_ID"])
            # width = 800
            # images = cv2.resize(images, (width, int((width / images.shape[1]) * images.shape[0])))

            key_points = model(images)
            key_points = torch.sigmoid(key_points)
            ftime += (time.time() - t)
            t = time.time()

            binary_kmap = key_points.squeeze().cpu().numpy() > line_detection_cfg['MODEL']['THRESHOLD']
            kmap_label = label(binary_kmap, connectivity=1)
            props = regionprops(kmap_label)
            plist = []
            for prop in props:
                plist.append(prop.centroid)

            size = (size[0][0], size[0][1])
            b_points = reverse_mapping(plist, numAngle=line_detection_cfg["MODEL"]["NUMANGLE"],
                                       numRho=line_detection_cfg["MODEL"]["NUMRHO"],
                                       size=(400, 400))
            scale_w = size[1] / 400
            scale_h = size[0] / 400
            for i in range(len(b_points)):
                y1 = int(np.round(b_points[i][0] * scale_h))
                x1 = int(np.round(b_points[i][1] * scale_w))
                y2 = int(np.round(b_points[i][2] * scale_h))
                x2 = int(np.round(b_points[i][3] * scale_w))
                if x1 == x2:
                    angle = -np.pi / 2
                else:
                    angle = np.arctan((y1 - y2) / (x1 - x2))
                (x1, y1), (x2, y2) = get_boundary_point(y1, x1, angle, size[0], size[1])
                b_points[i] = (y1, x1, y2, x2)

            vis = visulize_mapping(b_points, size[::-1], names[0])

            output_path = os.path.join(args.output_dir, names[0])
            cv2.imwrite(output_path, vis)
            np_data = np.array(b_points)
            np.save(os.path.join(args.output_dir, names[0].split('/')[-1].split('.')[0]), np_data)

            if line_detection_cfg["MODEL"]["EDGE_ALIGN"] and args.align:
                for i in range(len(b_points)):
                    b_points[i] = edge_align(b_points[i], names[0], size, division=5)
                vis = visulize_mapping(b_points, size, names[0])
                cv2.imwrite(join(args.output_image + '_align.png'), vis)
                np_data = np.array(b_points)
            ntime += (time.time() - t)

    # print('Forward time for total images: %.6f' % ftime)
    print('Line detection time for total images: %.6f' % ntime)
    return ftime + ntime


def save_centerline(read_path, output_path):
    # for file in glob.glob(os.path.join(read_path, '*')):
    for file in glob.glob(read_path + "*.jpg"):
        filename = os.path.split(file)[1]
        filename = os.path.splitext(filename)[0]

        # image = Image.open(file)
        img = cv2.imread(file)
        height = img.shape[0]
        width = img.shape[1]

        data = np.load(read_path + filename + '.npy')
        line1_p1 = data[0][0:2]  # y, x
        line1_p2 = data[0][2:4]
        line2_p1 = data[1][0:2]
        line2_p2 = data[1][2:4]

        if line1_p1[1] > line1_p2[1]:
            line1_x = line1_p2[1] + 1 / 3 * (line1_p1[1] - line1_p2[1])
        elif line1_p1[1] < line1_p2[1]:
            line1_x = line1_p2[1] - 1 / 3 * (line1_p2[1] - line1_p1[1])
        else:
            line1_x = line1_p1[1]

        if line2_p1[1] > line2_p2[1]:
            line2_x = line2_p2[1] + 1 / 3 * (line2_p1[1] - line2_p2[1])
        elif line2_p1[1] < line2_p2[1]:
            line2_x = line2_p2[1] - 1 / 3 * (line2_p2[1] - line2_p1[1])
        else:
            line2_x = line2_p1[1]

        centerline_p1 = np.array([2 * height / 3, 0.5 * (line1_x + line2_x)])
        centerline_p2 = np.array([height, 0.5 * (line1_p2[1] + line2_p2[1])])

        start_point = (int(centerline_p1[1]), int(centerline_p1[0]))
        end_point = (int(centerline_p2[1]), int(centerline_p2[0]))

        color = (0, 0, 255)
        thickness = 10

        cv2.line(img, start_point, end_point, color, thickness)
        cv2.imwrite(output_path + filename + '_cl.jpg', img)


if __name__ == '__main__':
    main()
