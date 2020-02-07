import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Dense,Activation,Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Model



IMG_SIZE = [224,224]


X = np.load('X_array_cone.npy')
Y = np.load('Y_array_cone.npy')

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.1)

mobile = MobileNetV2(input_shape = IMG_SIZE + [3], weights = 'imagenet',include_top = False)

for layer in mobile.layers:
	layer.trainable = False



x =Flatten()(mobile.output)

prediction = Dense(units = 1, activation = "sigmoid")(x)

model = Model(inputs = mobile.input, outputs = prediction)

model.summary()

model.compile(loss = "binary_crossentropy",optimizer = "adam", metrics = ["accuracy"])

model.fit(x_train,y_train, validation_data=(x_test,y_test), batch_size=16, epochs=5)

model.save("mobile_v2.h5")

model_json = model.to_json()
with open("mobilev2json.json", "w") as json_file:
    json_file.write(model_json)

print("model saved to disk")