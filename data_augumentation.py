from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
import numpy as np 


IMG_SIZE = 150

DATADIR1 = "/home/prahlad/cone"

image = load_img("/home/prahlad/cone/22.jpeg")          #86,5,76,25,60,56,9,91,92,93,2,3,4,30,41,50,22
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

total = 0

aug = ImageDataGenerator(featurewise_center = True,rotation_range = 20, width_shift_range = 0.2, height_shift_range = 0.2,horizontal_flip = True)
imageGen = aug.flow(image,batch_size = 1, save_to_dir = DATADIR1,save_prefix = "image", save_format = "jpg")

for image in imageGen:
	total += 1
	if total == 10:
		break

print("saved ")

