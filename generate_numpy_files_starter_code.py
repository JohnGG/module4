import numpy as np
import os
import cv2

# Define raw images paths
PATH_0 = "./dataset/0"
PATH_1 = "./dataset/1"

# Define image final sizes
SIZES = (224, 224)


# Create initial numpy variables to store data
X = []
Y = []

# Loop through class 0 dataset dir
for filename0 in os.listdir(PATH_0):
    # TODO : Store images image into X, label into Y
    pass

# Loop through class 0 dataset dir
for filename1 in os.listdir(PATH_1):
    # TODO : Store images image into X, label into Y
    pass

# Create a np index array for to shuffle the X and Y arrays
shuffle = np.arange(len(X))
np.random.shuffle(shuffle)

# TODO : Shuffle and split X and Y in 3 (train, val, test) and save it into .npy files
