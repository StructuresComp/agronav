#import libraries
import glob as glob
import os
import numpy as np

from PIL import Image
import cv2
# Import an image from directory:
read_path = 'data/inference/output/visualize_test/'
save_path = 'data/inference/centerline_visualized/'

# for file in glob.glob(os.path.join(read_path, '*')):
for file in glob.glob(read_path+"*.jpg"):
    filename = os.path.split(file)[1]
    filename = os.path.splitext(filename)[0]

    # image = Image.open(file)
    img = cv2.imread(file)
    height = img.shape[0]
    width = img.shape[1]

    data = np.load(read_path+filename+'.npy')
    line1_p1 = data[0][0:2] #y, x
    line1_p2 = data[0][2:4]
    line2_p1 = data[1][0:2]
    line2_p2 = data[1][2:4]

    if line1_p1[1] > line1_p2[1]:
        line1_x = line1_p2[1] + 1 / 3*( line1_p1[1]-line1_p2[1] )
    elif line1_p1[1] < line1_p2[1]:
        line1_x = line1_p2[1] - 1 / 3*( line1_p2[1]-line1_p1[1] )
    else:
        line1_x = line1_p1[1]

    if line2_p1[1] > line2_p2[1]:
        line2_x = line2_p2[1] + 1 / 3*( line2_p1[1]-line2_p2[1] )
    elif line2_p1[1] < line2_p2[1]:
        line2_x = line2_p2[1] - 1 / 3*( line2_p2[1]-line2_p1[1] )
    else:
        line2_x = line2_p1[1]

    centerline_p1 = np.array([2*height/3, 0.5*(line1_x + line2_x)])
    centerline_p2 = np.array([height, 0.5*(line1_p2[1] + line2_p2[1])])

    start_point = (int(centerline_p1[1]), int(centerline_p1[0]))
    end_point = (int(centerline_p2[1]), int(centerline_p2[0]))

    color = (0, 0, 255)
    thickness = 10

    cv2.line(img, start_point, end_point, color, thickness)
    cv2.imwrite(save_path+filename+'_cl.jpg', img)
    # cv2.imshow('test', img)
    # image = Image.open(file)
    # resized_image = image.resize((400,400))
    #
    # filename = os.path.split(file)[1]
    # filename = os.path.splitext(filename)[0]
    #
    # resized_image.save(save_path+filename+'.jpg')
