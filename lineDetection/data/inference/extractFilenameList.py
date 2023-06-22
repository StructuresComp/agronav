#import libraries
import glob as glob
import os
import numpy as np

from PIL import Image

# Import an image from directory:

read_path = './data/inference/input/'

with open('./data/inference/test_idx.txt', 'w') as f:
    for file in glob.glob(os.path.join(read_path, '*.jpg')):
	
        filename = os.path.split(file)[1]
        filename = os.path.splitext(filename)[0]
   
        f.write('input/'+filename+"\n")
  
