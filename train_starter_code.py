import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
# TODO : Choose your hyper parameters
epochs = None
batch_size = None

# Read dataset
# TODO : replace with your paths
X_TRAIN = np.load("../X_TRAIN.npy")
Y_TRAIN = np.load("../Y_TRAIN.npy")
X_VAL = np.load("../X_VAL.npy")
Y_VAL = np.load("../Y_VAL.npy")



###### MODEL DEFINITION #####

# Define placholders
# TODO : Replace with the shape you decided to create for inputs
inputs = tf.placeholder(shape=(None, None, None), name="inputs", dtype=tf.float32)
labels = tf.placeholder(shape=(None, ), name="labels", dtype=tf.int64)

#TODO :  Define Your model

#TODO :  Create loss node

#TODO :  Define optimizer ops

#TODO : define accuracy

# Create saver var to save model
saver = tf.train.Saver(max_to_keep=1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for e in range(epochs):
        ###### TRAINING LOOP #####
        # TODO: Mini batch gradient descent (batch size = 500)
        # TODO: Compute accuracy and loss for train dataset after all the optimisation execution are done
        ###### /TRAINING LOOP #####


        ###### EVALUATION LOOP #####
        # TODO: Compute accuracy and loss for val dataset
        # TODO: Save the model if it goes over 90% accuracy, then break
        # Example for saving : save_path = saver.save(sess, "model.ckpt")
        ###### /EVALUATION LOOP #####
        pass