import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

IMG_SIZE = 224


X = np.load('X_array.npy')
Y = np.load('Y_array.npy')

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.1)

learning_rate = 0.001
num_steps = 200
batch_size = 16


num_input = 784
num_classes = 1
dropout = 0.00005
def conv_net(x_train,num_classes,dropout,reuse,is_training):
	with tf.variable_scope('ConvNet',reuse=reuse):

		conv1 = tf.layers.conv2d(x_train,64,3,activation = tf.nn.relu,padding = "same")
		conv2 = tf.layers.conv2d(conv1,64,3,activation = tf.nn.relu,padding = "same")
		maxpool1 = tf.layers.max_pooling2d(conv2,2,2)
		conv3 = tf.layers.conv2d(maxpool1,128,3,activation = tf.nn.relu,padding = "same")
		conv4 = tf.layers.conv2d(conv3,128,1,activation = tf.nn.relu,padding = "same")
		conv5 = tf.layers.conv2d(conv4,128,1,activation = tf.nn.relu,padding = "same")
		maxpool2 = tf.layers.max_pooling2d(conv5,2,2)
		conv6 = tf.layers.conv2d(maxpool2,256,3,activation = tf.nn.relu,padding = "same")
		conv7 = tf.layers.conv2d(conv6,256,3,activation = tf.nn.relu,padding = "same")
		conv8 = tf.layers.conv2d(conv7,256,3,activation = tf.nn.relu,padding = "same") 
		conv9 = tf.layers.conv2d(conv8,512,1,activation = tf.nn.relu,padding = "same")
		maxpool3 = tf.layers.max_pooling2d(conv9,2,2) 
		conv10 = tf.layers.conv2d(maxpool3,512,3,activation = tf.nn.relu,padding = "same")
		conv11 = tf.layers.conv2d(conv10,512,3,activation = tf.nn.relu,padding = "same")
		conv12 = tf.layers.conv2d(conv11,512,3,activation = tf.nn.relu,padding = "same")
		maxpool4 = tf.layers.max_pooling2d(conv12,2,2)
		conv13 = tf.layers.conv2d(maxpool4,512,3,activation = tf.nn.relu,padding = "same")
		conv14 = tf.layers.conv2d(conv13,512,3,activation  = tf.nn.relu,padding = "same")
		conv15 = tf.layers.conv2d(conv14,512,3,activation = tf.nn.relu,padding = "same")
		maxpool5 = tf.layers.max_pooling2d(conv15,2,2)
		fc1 = tf.contrib.layers.flatten(maxpool5)
		fc2 = tf.layers.dense(fc1,4096,activation = tf.nn.relu)
		fc3 = tf.layers.dense(fc2,4096,activation = tf.nn.relu)
		fc4 = tf.layers.dropout(fc3, rate = dropout, training = is_training)
		out = tf.layers.dense(fc4,num_classes)
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







