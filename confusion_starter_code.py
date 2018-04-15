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
inputs = tf.placeholder(shape=(None, Noe, Noe, Noe), name="inputs", dtype=tf.float32)
labels = tf.placeholder(shape=(None, ), name="labels", dtype=tf.int64)

# TODO: Paste your own model, you only need to go to the prediction node

# Create saver var to restore model
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # TODO : Replace with your ckpt file path
    saver.restore(sess, "./saved/model.ckpt")

    for idx, x in enumerate(X_TEST):
        pred = sess.run(predictions, feed_dict={inputs: [X_TEST[idx]], labels: [Y_TEST[idx]]})
        # TODO: Compute the percentage of false negatives and false positives