import random
import os
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
IMG_SIZE = 400

DATADIR1 = "/home/yash/AeroMit/VTOL_Sim/CNN/1"


DATADIR0 = "/home/yash/AeroMit/VTOL_Sim/CNN/0"

count = 0

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


num_input = 160000
num_classes = 1
dropout = 0.25

def conv_net(x_train,num_classes,dropout,reuse,is_training):
	with tf.variable_scope('ConvNet',reuse=reuse):
		conv1 = tf.layers.conv2d(x_train,64,3,activation = tf.nn.relu)
		conv1 = tf.layers.max_pooling2d(conv1,2,2)
		conv1 = tf.layers.batch_normalization(conv1)

		conv2 = tf.layers.conv2d(conv1,32,3,activation = tf.nn.relu)
		conv2 = tf.layers.max_pooling2d(conv2,2,2)
		conv2 = tf.layers.batch_normalization(conv2)


		conv3 = tf.layers.conv2d(conv2,16,3,activation = tf.nn.relu)
		conv3 = tf.layers.max_pooling2d(conv3,2,2)
		conv3 = tf.layers.batch_normalization(conv3)


		conv4 = tf.layers.conv2d(conv3,8,3,activation = tf.nn.relu)
		conv4 = tf.layers.max_pooling2d(conv4,2,2)
		conv4 = tf.layers.batch_normalization(conv4)

		fc1 = tf.contrib.layers.flatten(conv4)

		fc1 = tf.layers.dense(fc1,1024)

		fc1 = tf.layers.dropout(fc1, rate = dropout, training = is_training)

		out = tf.layers.dense(fc1,num_classes)
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


input_fn = tf.estimator.inputs.numpy_input_fn(x = x_train, y=y_train,batch_size=batch_size, num_epochs=15, shuffle=True)


model.train(input_fn, steps=num_steps)

input_fn = tf.estimator.inputs.numpy_input_fn(x=x_test, y=y_test,batch_size=batch_size, shuffle=False)
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])





