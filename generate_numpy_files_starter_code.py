import cv2
import numpy as np
import os

# Get number of files in each class
TOTAL_0 = len(os.listdir("../datasets/expenses/0"))
TOTAL_1 = len(os.listdir("../datasets/expenses/1"))

# Store dataset paths to variables
# TODO: Replace with your real paths
PATH_0 = "../datasets/expenses/0/"
PATH_1 = "../datasets/expenses/1/"

# Create initial arrays to store data
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
