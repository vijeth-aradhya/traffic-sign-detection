"""
To run:

python crop.py <training images folder>

Example: python crop.py ./GTSRB/Final_Training/Images/

Returns: A cropped and resized jpg version of the .ppm files
"""

import os
import sys
import math

import pandas as pd
import numpy as np
import cv2

train_dir = sys.argv[1]

folder_list = os.listdir(train_dir)

max_width = max_height = -1
min_width = min_height = 1000000

for folder in folder_list:
    files = os.listdir(train_dir + folder)
    csv = [fil for fil in files if 'csv' in fil]
    csv_file = train_dir + folder + '/' + csv[0]
    df = pd.read_csv(csv_file, header=None)
    data = df.iloc[1:, :].values
    for row in data:
        image_row = row[0].split(';')
        image = image_row[0]
        root = image[:-4]
        im = cv2.imread(train_dir + folder + '/' + image)
        im = im[int(image_row[4]):int(image_row[6]), int(image_row[3]):int(image_row[5])]
        cv2.imwrite(train_dir + folder + '/' + root + '.jpg', im)
        height = math.fabs(int(image_row[6]) - int(image_row[4]))
        width = math.fabs(int(image_row[5]) - int(image_row[3]))
        if width < min_width:
            min_width = width
        elif width > max_width:
            max_width = width
        if height < min_height:
            min_height = height
        elif height > max_height:
            max_height = height

        """
        To shrink an image, it will generally look best with CV_INTER_AREA interpolation,
        whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC (slow)
        or CV_INTER_LINEAR (faster but still looks OK).

        resized_img = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
        resized_img = cv2.resize(im, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        """
# print min_width, min_height
# print max_width, max_height

scale_width = (min_width + max_width)/float(2)
scale_height = (min_height + max_height)/float(2)
# print scale_width, scale_height

for folder in folder_list:
    files = os.listdir(train_dir + folder)
    images = [fil for fil in files if '.jpg' in fil]
    for image in images:
        root = image[:-4]
        im = cv2.imread(train_dir + folder + '/' + image)
        height = np.size(im, 0)
        width = np.size(im, 1)
        fx = float(scale_width)/width
        fy = float(scale_height)/height
        if fx < float(1) or fy < float(1):
            resized_img = cv2.resize(im, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
        else:
            resized_img = cv2.resize(im, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(train_dir + folder + '/' + root + '_resized.jpg', resized_img)
