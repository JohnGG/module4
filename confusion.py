import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Read dataset
# TODO : replace with your paths
X_TEST = np.load("./X_TEST.npy")
Y_TEST = np.load("./Y_TEST.npy")



###### MODEL DEFINITION #####

# Define placholders
# TODO : Replace with the shape you decided to create for inputs
inputs = tf.placeholder(shape=(None, 80, 80, 3), name="inputs", dtype=tf.float32)
labels = tf.placeholder(shape=(None, ), name="labels", dtype=tf.int64)
conv1 = tf.layers.conv2d(
      inputs=tf.reshape(inputs, shape=(-1, 80, 80, 3)),
      filters=32,
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=32,
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
pool2_shapes = pool2.get_shape().as_list()
W = tf.get_variable(name="weights", shape=(pool2_shapes[1]*pool2_shapes[2]*pool2_shapes[3], 2))
B = tf.get_variable(name="bias2", shape=(2))
logits = tf.matmul(tf.reshape(pool2, shape=(-1, pool2_shapes[1]*pool2_shapes[2]*pool2_shapes[3])), W) + B
predictions = tf.argmax(tf.nn.softmax(logits), axis=1)

# Create saver var to restore model
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./saved/model.ckpt")
    false_positives = 0
    false_negatives = 0
    good_predictions = 0
    for idx, x in enumerate(X_TEST):
        pred = sess.run(predictions, feed_dict={inputs: [X_TEST[idx]], labels: [Y_TEST[idx]]})
        if pred[0] == 1 and Y_TEST[idx] == 0:
            false_positives+=1
            cv2.imshow("False positive", X_TEST[idx])
            cv2.waitKey(0)
        elif pred[0] == 0 and Y_TEST[idx]  == 1:
            false_negatives+=1
            cv2.imshow("False negative" ,X_TEST[idx])
            cv2.waitKey(0)
        elif pred[0] == Y_TEST[idx]:
            good_predictions += 1
    print("Accuracy : %f" % float(good_predictions/len(X_TEST)))
    print("False positives : %f" % float(false_positives/len(X_TEST)))
    print("False negatives : %f" % float(false_negatives/len(X_TEST)))