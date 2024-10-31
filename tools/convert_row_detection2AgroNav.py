import json
import tkinter as tk
from tkinter import filedialog
from tkinter.font import nametofont
import numpy as np
import os
import pandas as pd
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Row Detection jsons labels to agronav's format")
    parser.add_argument('--input_folder', type=str, default="/home/r4hul-lcl/Datasets/row-detection-agronav/test", help='Path to the input folder containing JSON files')
    parser.add_argument('--labels_folder', type=str, default="labels", help='labels folder name')
    parser.add_argument('--output_folder', type=str, default="labels-agronav", help='folder name for outputs')
    parser.add_argument('--dont_delete', type=bool, default=True, help='dont delete')
    return parser.parse_args()


def get_boundary_point(y_pnts, x_pnts, pnt, W):
    p_y = np.polyfit(y_pnts, x_pnts, N_DEGREE)
    p_x = np.polyfit(x_pnts, y_pnts, N_DEGREE)
    x_close = np.polyval(p_y, pnt)
    y_close = pnt
    if x_close < 0 or x_close > W:
        x_close = 0 if x_close < 0 else W
        y_close = np.polyval(p_x, x_close)
    return x_close, y_close

def get_cords(data_df, idxs_l, idxs_r, H, W):
    coords = []
    for idx in [idxs_l, idxs_r]:
        x_pnts = np.array(data_df.iloc[idx]['x'])
        y_pnts = np.array(data_df.iloc[idx]['y'])
        y_idxs = np.argsort(y_pnts)
        x_pnts = x_pnts[y_idxs]
        y_pnts = y_pnts[y_idxs]
        x1, y1 = get_boundary_point(y_pnts, x_pnts, 0, W)
        coords.append(int(x1))
        coords.append(int(y1))
        x2, y2 = get_boundary_point(y_pnts, x_pnts, H, W)
        coords.append(int(x2))
        coords.append(int(y2))
    return coords

def find_ego_idx(key_points, fileName, H, W):
    dist_x = []
    idxs = []
    l_val = 0
    r_val = 0
    for idx, key_point in key_points.iterrows():
        p_y = np.polyfit(key_point['y'], key_point['x'], N_DEGREE)
        x_close = np.polyval(p_y, H)
        dist_x.append(x_close-0.5*W)
        idxs.append(idx)
    dist_x = np.array(dist_x)
    idxs = np.array(idxs)
    idxs_l = idxs[dist_x < 0]#idxs[dist_x > 0]
    xs_l = dist_x[dist_x < 0]#dist_x[dist_x > 0]
    
    idxs_r = idxs[dist_x >= 0]#idxs[dist_x <= 0]
    xs_r = dist_x[dist_x >= 0]#dist_x[dist_x <= 0]

    if(len(xs_r) == 0):
        # with open("/home/deleted_files.txt", 'w') as txtFile:
        #         txtFile.write(str(fileName))
        #         txtFile.close()

        with open("temp.txt", 'a') as txtFile:
            txtFile.write("\n" + str(fileName))
            txtFile.close()

        return 100, 100
    elif(len(xs_l) == 0):
        # with open("/home/deleted_files.txt", 'w') as txtFile:
        #         txtFile.write(str(fileName))
        #         txtFile.close()

        with open("temp.txt", 'a') as txtFile:
            txtFile.write("\n" + str(fileName))
            txtFile.close()

        return 100, 100
    else:
        return idxs_l[np.argmax(xs_l)], idxs_r[np.argmin(xs_r)]

def main():

    args = parse_args()
    input_folder_path = args.input_folder
    labels_folder = os.path.join(input_folder_path, args.labels_folder)
    images_folder = os.path.join(input_folder_path, "images")
    output_folder_path = os.path.join(input_folder_path, args.output_folder)
    progress = 0
    total_files = len(os.listdir(labels_folder))
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    for file in os.listdir(labels_folder): 
        if(file.endswith(".json")): 
            image_path = file.replace(".json", ".jpg")  
            image_path = os.path.join(images_folder, image_path)
            coords = []
            img = cv2.imread(image_path, cv2.IMREAD_COLOR) 
            if img is None:
                print("Image not found")
                continue
            with open(os.path.join(labels_folder, file)) as f:
                data = json.load(f)
                data_df = pd.json_normalize(data, 'labels')
                f.close()


            H, W = img.shape[:2]
            idxs_l, idxs_r = find_ego_idx(data_df, file, H, W)

            if(idxs_l == 100 and idxs_r == 100):
                if not args.dont_delete:
                    os.remove(os.path.join(labels_folder, file))
                    os.remove(image_path)
                else:
                    with open("bad_files.txt", 'a') as txtFile:
                        txtFile.write("\n" + str(file))
                pass

            else:



                coords = get_cords(data_df, idxs_l, idxs_r, H, W)
                cv2.line(img, coords[0:2], coords[2:4], (0,0,255), 5)
                cv2.line(img, coords[4:6], coords[6:8], (0,0,255), 5)
                # Display the image
                label_path = output_folder_path + '/' + file.replace('.json', '.txt')

                with open(label_path, 'w') as f:
                    f.write(f"{2}")
                    for coord in coords:
                        f.write(f" {coord}")
                    f.close()

                image_out_path = label_path.replace('.txt', '.jpg')
                cv2.imwrite(image_out_path, img)
                # cv2.imshow('Image with Lines', img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        progress +=1

if __name__ == "__main__":
    N_DEGREE = 1
    main()

            

# def draw_line(y, x, angle, image, color=(0,0,255), num_directions=24):
#     '''
#     Draw a line with point y, x, angle in image with color.
#     '''
#     cv2.circle(image, (x, y), 2, color, 2)
#     H, W = image.shape[:2]
#     angle = int2arc(angle, num_directions)
#     point1, point2 = get_boundary_point(y, x, angle, H, W)
#     cv2.line(image, point1, point2, color, 2)
#     return image

