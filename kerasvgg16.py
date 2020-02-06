import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import keras 
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

IMG_SIZE = 228


X = np.load('X_array.npy')
Y = np.load('Y_array.npy')


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.1)
model = Sequential()
model.add(Conv2D(input_shape=(IMG_SIZE,IMG_SIZE,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2),padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2),padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2),padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2),padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2),padding = 'same'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=2, activation="relu"))



model.add(Dropout(0))



model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.summary()
model.fit(x_train,y_train, validation_data=(x_test,y_test), batch_size=16, epochs=5)




#model.save("blue_ball_test.h5")

#model_json = model.to_json()
#with open("model_test.json1", "w") as json_file:
    #json_file.write(model_json)

#print("model saved to disk")
