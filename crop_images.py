"""
To run:

python crop.py <training images folder>

Example: python crop.py ./GTSRB/Final_Training/Images/

Returns: A cropped jpg version of the .ppm files
"""

import os
import sys

import pandas as pd
import cv2

train_dir = sys.argv[1]

folder_list = os.listdir(train_dir)

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
        # cv2.imshow("original", im)
        # cv2.waitKey(0)
        # im = im[y1:y2, x1:x2] ; (x1,y1) as the top-left vertex and (x2,y2) as the bottom-right 
        im = im[int(image_row[4]):int(image_row[6]), int(image_row[3]):int(image_row[5])]
        # im.save(train_dir + folder + '/' + root + '.jpg')
        """
        To shrink an image, it will generally look best with CV_INTER_AREA interpolation,
        whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC (slow)
        or CV_INTER_LINEAR (faster but still looks OK).

        resized_img = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
        resized_img = cv2.resize(im, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        """
