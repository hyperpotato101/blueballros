import os
import random
import cv2
import numpy as np
import glob

IMG_SIZE = 28

DATADIR1 = "/home/prahlad/Pictures/1"


DATADIR0 = "/home/prahlad/Pictures/0"


training_data = []
def create_training_data_1():
	global count
	data_path = os.path.join(DATADIR1,'*g')
	files = glob.glob(data_path)
	for f in files:
		img_array = cv2.imread(f,cv2.IMREAD_COLOR)
		resized_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
		training_data.append([resized_array,1])
	count = len(training_data)
	print(count)


def create_training_data_0():
	global count
	data_path = os.path.join(DATADIR0,'*g')
	files = glob.glob(data_path)
	for f in files:
		img_array = cv2.imread(f,cv2.IMREAD_COLOR)
		resized_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
		training_data.append([resized_array,0])
	print(len(training_data) - count)


create_training_data_1()
create_training_data_0()
random.shuffle(training_data)


X = []
Y = []


for features, label in training_data:
	X.append(features)
	Y.append(label)
Y = np.asarray(Y).astype('float32').reshape((-1,1))
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE,3)
X = X/255.0

X = np.array(X)

X = np.float32(X)

np.save('X_array',X)

print('X_array saved to disk')

Y = np.array(Y)
Y= np.float32(Y)

np.save('Y_array',Y)


print('Y_array saved to disk')










