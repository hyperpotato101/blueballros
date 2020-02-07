import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Dense,Activation,Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model




IMG_SIZE = [224,224]


X = np.load('X_array_cone.npy')
Y = np.load('Y_array_cone.npy')

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.1)



	

vgg = VGG16(input_shape = IMG_SIZE + [3], weights = 'imagenet',include_top = False)

for layer in vgg.layers:
	layer.trainable = False



x =Flatten()(vgg.output)

prediction = Dense(units = 1, activation = "sigmoid")(x)

model = Model(inputs = vgg.input, outputs = prediction)

model.summary()

model.compile(loss = "binary_crossentropy",optimizer = "adam", metrics = ["accuracy"])

model.fit(x_train,y_train, validation_data=(x_test,y_test), batch_size=16, epochs=1)

model.save("vgg16.h5")

model_json = model.to_json()
with open("vgg16json.json", "w") as json_file:
    json_file.write(model_json)

print("model saved to disk")