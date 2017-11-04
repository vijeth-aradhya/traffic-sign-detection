"""
To run:

python crop.py <training images folder>

Example: python crop.py ./GTSRB/Final_Training/Images/

Returns: A cropped jpg version of the .ppm files
"""

from PIL import Image
import os, sys
import pandas as pd

train_dir = sys.argv[1]

folder_list = os.listdir(train_dir)
print folder_list
# im = Image.open("00000_00000.ppm")
# im2 = im.crop((5, 6, 24, 25))
# im2.save("Out.jpg")
for folder in folder_list:
    print folder
    files = os.listdir(train_dir + folder)
    csv = [fil for fil in files if 'csv' in fil]
    # print csv
    csv_file = train_dir + folder + '/' + csv[0]
    # print csv_file
    df = pd.read_csv(csv_file,header=None)
    data = df.iloc[1:, :].values
    for row in data:
        image_row = row[0].split(';')
        image = image_row[0]
        root = image[:-4]
        print image, root
        im = Image.open(train_dir + folder + '/' + image)
        im = im.crop((int(image_row[3]),int(image_row[4]),int(image_row[5]),int(image_row[6])))
        im.save(train_dir + folder + '/' + root +'.jpg')
        # break
    # break
