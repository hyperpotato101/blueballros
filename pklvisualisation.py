from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.models import Model
import keras
from pickle import dump


image = load_img('/home/prahlad/Pictures/cone_1/5.jpeg', target_size=(224,224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)
model = keras.applications.vgg16.VGG16()
model.layers.pop()    #removing the last layer of model
model = Model(inputs = model.inputs,outputs = model.layers[-1].output) #output from the pre trained model will be one layer less here it will be a 4096 coloumn tensor
features = model.predict(image)
print(features.shape)

dump(features, open('dog.pkl', 'wb'))



