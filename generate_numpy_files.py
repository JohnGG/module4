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

# Create initial numpy variables to store data
X = []
Y = []

# Loop through class 0 dataset dir
for filename0 in os.listdir(PATH_0):
    # TODO : Store images image into X, label into Y
    X.append(cv2.imread(PATH_0+filename0))
    Y.append(0)
    print(filename0)

# Loop through class 0 dataset dir
for filename1 in os.listdir(PATH_1):
    # TODO : Store images image into X, label into Y
    X.append(cv2.imread(PATH_1+filename1))
    Y.append(1)
    print(filename1)

# Create a np index array for to shuffle the X and Y arrays
shuffle = np.arange(len(X))
np.random.shuffle(shuffle)

# TODO : Shuffle and split X and Y in 3 (train, val, test) and save it into .npy files
X = np.array(X)[shuffle]
Y = np.array(Y)[shuffle]

X_TRAIN = X[0:int(len(X)*0.8)]
Y_TRAIN = Y[0:int(len(X)*0.8)]

X_VAL = X[int(len(X)*0.8):int(len(X)*0.9)]
Y_VAL = Y[int(len(X)*0.8):int(len(X)*0.9)]

X_TEST = X[int(len(X)*0.9):int(len(X))]
Y_TEST = Y[int(len(X)*0.9):int(len(X))]

np.save("X_TRAIN.npy", X_TRAIN)
np.save("Y_TRAIN.npy", Y_TRAIN)

np.save("X_VAL.npy", X_VAL)
np.save("Y_VAL.npy", Y_VAL)

np.save("X_TEST.npy", X_TEST)
np.save("Y_TEST.npy", Y_TEST)
