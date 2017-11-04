"""
To run:

python crop.py <training images folder>

Example: python crop.py ./GTSRB/Final_Training/Images/

Returns: A cropped jpg version of the .ppm files
"""

import os
import sys

import pandas as pd
from PIL import Image

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
        im = Image.open(train_dir + folder + '/' + image)
        im = im.crop((int(image_row[3]), int(image_row[4]), int(image_row[5]), int(image_row[6])))
        im.save(train_dir + folder + '/' + root + '.jpg')
