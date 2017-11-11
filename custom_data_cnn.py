
import os,cv2, sys, math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras import backend as K
K.set_image_dim_ordering('th')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam

#%%
num_channel=3
num_epoch=3

# Define the number of classes
num_classes = 43

img_data_list=[]

train_dir = sys.argv[1]

folder_list = os.listdir(train_dir)
folder_list.sort()

max_width = max_height = -1
min_width = min_height = 1000000

num_images = 0
print( 'Cropping Starts')
print( time.time())

number_files = []
for folder in folder_list:
	print(train_dir)
	print(folder)
	num_images = 0
	files = os.listdir(train_dir + '/'+ folder)
	files.sort()
	number_files.append('200')
	csv = [fil for fil in files if 'csv' in fil]
	csv_file = train_dir + '/' + folder + '/' + csv[0]
	df = pd.read_csv(csv_file, header=None)
	data = df.iloc[1:, :].values
	for row in data:
		print (num_images)
		if num_images >= 200:
			continue
		else:
			num_images += 1
			image_row = row[0].split(';')
			image = image_row[0]
			root = image[:-4]
			im = cv2.imread(train_dir + '/' + folder + '/' + image)
			im = im[int(image_row[4]):int(image_row[6]), int(image_row[3]):int(image_row[5])]
			cv2.imwrite(train_dir + '/' + folder + '/' + root + '.jpg', im)
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
print ('Cropping Ends')
print (time.time())


print ('Scaling Starts')
num_images = 0
for folder in folder_list:
	print (folder)
	files = os.listdir(train_dir + '/' + folder)
	images = [fil for fil in files if '.jpg' in fil]
	images.sort()
	print (len(images))
	num_images = 0
	for image in images:
		root = image[:-4]
		im = cv2.imread(train_dir + '/' + folder + '/' + image)
		height = np.size(im, 0)
		width = np.size(im, 1)
		fx = float(scale_width)/width
		fy = float(scale_height)/height
		if fx < float(1) or fy < float(1):
			resized_img = cv2.resize(im, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
			img_data_list.append(resized_img)
		else:
			resized_img = cv2.resize(im, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
			img_data_list.append(resized_img)
		cv2.imwrite(train_dir + '/' + folder + '/' + root + '_resized.jpg', resized_img)


img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)

print ('Scaling ENds')
print (time.time())



if num_channel==1:
	if K.image_dim_ordering()=='th':
		img_data= np.expand_dims(img_data, axis=1) 
		print (img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=4) 
		print (img_data.shape)
		
else:
	if K.image_dim_ordering()=='th':
		img_data=np.rollaxis(img_data,3,1)
		print (img_data.shape)
		
print ('Rollaxis ends')
print (time.time())

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

start = 0
for i in range(len(number_files)):
	if i == len(number_files) - 1:
		labels[start:] = i
		print(i)
	else:
		labels[start:int(number_files[i])] = i
		print(i)
		start = int(number_files[i])
names = [str(i) for i in range(43)]

# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

#%%
# Defining the model
input_shape=img_data[0].shape
		

print ('model starts building')
print (time.time())		
model = Sequential()

model.add(Convolution2D(32, 3,3,border_mode='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
#model.add(Convolution2D(64, 3, 3))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])

print( 'model built and compiled')
print (time.time())

# Training
hist = model.fit(X_train, y_train, batch_size=16, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test))
print ('model fitted')
print (time.time())

print('Starting testing')
print( time.time())
test_dir = sys.argv[2]

image_list = os.listdir(test_dir)
image_list.sort()
csv_file = [fil for fil in image_list if '.csv' in fil]
images = [im for im in image_list if '.ppm' in im]
df = pd.read_csv(test_dir + csv_file[0], header=None)
data = df.iloc[1:,:].values
# Testing a new image

for row in data:
	image_row = row[0].split(';')
	image = image_row[0]
	print(image)
	root = image[:-4]
	im = cv2.imread(test_dir + '/' + image)
	im = im[int(image_row[4]):int(image_row[6]), int(image_row[3]):int(image_row[5])]
	height = math.fabs(int(image_row[6]) - int(image_row[4]))
	width = math.fabs(int(image_row[5])- int(image_row[3]))
	fx = float(scale_width)/width
	fy = float(scale_height)/height
	resized_img = ''
	if fx < float(1) or fy < float(1):
		resized_img = cv2.resize(im, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)	
	else:
		resized_img = cv2.resize(im, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)	
	test_image = resized_img
	test_image = np.array(test_image)
	test_image = test_image.astype('float32')
	test_image /= 255
	if num_channel==1:
		if K.image_dim_ordering()=='th':
			test_image= np.expand_dims(test_image, axis=0)
			test_image= np.expand_dims(test_image, axis=0)
			print (test_image.shape)
		else:
			test_image= np.expand_dims(test_image, axis=3) 
			test_image= np.expand_dims(test_image, axis=0)
			print (test_image.shape)
		
	else:
		if K.image_dim_ordering()=='th':
			test_image=np.rollaxis(test_image,2,0)
			test_image= np.expand_dims(test_image, axis=0)
			print (test_image.shape)
		else:
			test_image= np.expand_dims(test_image, axis=0)
			print (test_image.shape)
		
	# Predicting the test image
	print((model.predict(test_image)))
	s1 = model.predict_classes(test_image)
	print(s1)
	s2 = image_row[7]
	print(s2)
	with open('out.txt','w+') as f:
		f.write(str(s1) + str(s2) + '\n')
	f.close()

print( 'Testing Ends')
print (time.time())

