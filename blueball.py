# created on 20 jan 2019 by prahalad
import numpy as np
import cv2
import os
import glob
import keras 
import random
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split

IMG_SIZE = 150

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
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE,3)
X = X/255.0


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.1)
model = Sequential()
model.add(Convolution2D(128,3,strides= (2,2), padding='same',data_format="channels_last",activation='relu',input_shape=(IMG_SIZE,IMG_SIZE,3)))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2),padding = 'same'))
model.add(BatchNormalization())
model.add(Convolution2D(128,3,strides=(2,2),padding='same',data_format="channels_last",activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2),padding = 'same'))
model.add(BatchNormalization())
model.add(Convolution2D(256,3,strides=(2,2),padding='same',data_format="channels_last",activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2),padding = 'same'))
model.add(BatchNormalization())
model.add(Convolution2D(256,3,strides=(2,2),padding='same',data_format="channels_last",activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2),padding = 'same'))
model.add(BatchNormalization())
model.add(Convolution2D(1024,3,strides=(2,2),padding='same',data_format="channels_last",activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2),padding = 'same'))
model.add(BatchNormalization())
model.add(Flatten())


model.add(Dense(units = 1024, activation = 'relu'))


model.add(Dropout(0))



model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.summary()
model.fit(x_train,y_train, validation_data=(x_test,y_test), batch_size=16, epochs=10)




#model.save("blue_ball_test.h5")

#model_json = model.to_json()
#with open("model_test.json1", "w") as json_file:
    #json_file.write(model_json)

#print("model saved to disk")
