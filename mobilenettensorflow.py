import random
import os
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
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

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.1)

learning_rate = 0.001
num_steps = 200
batch_size = 16


num_input = 784
num_classes = 1
dropout = 0.0
def conv_net(x_train,num_classes,dropout,reuse,is_training):
	with tf.variable_scope('ConvNet',reuse=reuse):

		conv1 = tf.layers.conv2d(x_train,32,3,activation = tf.nn.relu,padding = "same")
		conv2 = tf.layers.conv2d(conv1,32,1,activation = tf.nn.relu,padding = "same")
		conv3 = tf.layers.conv2d(conv2,64,1,activation = tf.nn.relu,padding = "same")
		conv4 = tf.layers.conv2d(conv3,128,1,activation = tf.nn.relu,padding = "same")
		conv5 = tf.layers.conv2d(conv4,128,1,activation = tf.nn.relu,padding = "same")
		conv6 = tf.layers.conv2d(conv5,256,1,activation = tf.nn.relu,padding = "same")
		conv7 = tf.layers.conv2d(conv6,256,1,activation = tf.nn.relu,padding = "same")
		conv8 = tf.layers.conv2d(conv7,512,1,activation = tf.nn.relu,padding = "same") 
		conv9 = tf.layers.conv2d(conv8,512,1,activation = tf.nn.relu,padding = "same") 
		conv10 = tf.layers.conv2d(conv9,512,1,activation = tf.nn.relu,padding = "same")
		conv11 = tf.layers.conv2d(conv10,512,1,activation = tf.nn.relu,padding = "same")
		conv12 = tf.layers.conv2d(conv11,512,1,activation = tf.nn.relu,padding = "same")
		conv13 = tf.layers.conv2d(conv12,512,1,activation = tf.nn.relu,padding = "same")
		conv14 = tf.layers.conv2d(conv13,1024,1,activation  = tf.nn.relu,padding = "same")
		average_pool = tf.layers.average_pooling2d(conv14,7,7,padding = "same")
		fc1 = tf.contrib.layers.flatten(average_pool)
		fc2 = tf.layers.dense(fc1,1024)
		fc3 = tf.layers.dropout(fc2, rate = dropout, training = is_training)
		out = tf.layers.dense(fc3,num_classes)
	return out



def model_fn(features, labels, mode):
	logits_train = conv_net(features,num_classes,dropout,reuse = False,is_training = True)
	logits_test = conv_net(features,num_classes,dropout,reuse = True,is_training = False)
	

	pred_classes = tf.argmax(logits_test,axis = 1)
	pred_probas = tf.nn.sigmoid(logits_test)


	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

	loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_train, labels=tf.cast(labels, dtype=tf.float32)))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

	acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

	estim_specs = tf.estimator.EstimatorSpec(mode=mode,predictions=pred_classes,loss=loss_op,train_op=train_op,eval_metric_ops={'accuracy': acc_op})

	return estim_specs

model = tf.estimator.Estimator(model_fn)


input_fn = tf.estimator.inputs.numpy_input_fn(x = x_train, y=y_train,batch_size=batch_size, num_epochs=5, shuffle=True)


model.train(input_fn, steps=num_steps)

input_fn = tf.estimator.inputs.numpy_input_fn(x=x_test, y=y_test,batch_size=batch_size, shuffle=False)
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])







