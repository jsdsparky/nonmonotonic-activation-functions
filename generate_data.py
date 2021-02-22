# Jack DeLano

import numpy as np
from sklearn.model_selection import train_test_split


NUM_GAUSSIANS_PER_CLASS = 100000
DATA_DIM = 25
GAUSSIAN_MEAN_MAX_L_INF_NORM = 1
GAUSSIAN_COV_MIN = 0.01
GAUSSIAN_COV_MAX = 0.1
NUM_SAMPLES = 500000

# Randomly generate data distributions
redMeans = (np.random.random((NUM_GAUSSIANS_PER_CLASS, DATA_DIM)) - 0.5) * GAUSSIAN_MEAN_MAX_L_INF_NORM
blueMeans = (np.random.random((NUM_GAUSSIANS_PER_CLASS, DATA_DIM)) - 0.5) * GAUSSIAN_MEAN_MAX_L_INF_NORM

redCovs = np.zeros((NUM_GAUSSIANS_PER_CLASS, DATA_DIM, DATA_DIM))
blueCovs = np.zeros((NUM_GAUSSIANS_PER_CLASS, DATA_DIM, DATA_DIM))
for i in range(NUM_GAUSSIANS_PER_CLASS):
    redCovs[i] = np.diag(np.random.random(DATA_DIM) * (GAUSSIAN_COV_MAX - GAUSSIAN_COV_MIN) + GAUSSIAN_COV_MIN)
    redCovs[i] = np.diag(np.random.random(DATA_DIM) * (GAUSSIAN_COV_MAX - GAUSSIAN_COV_MIN) + GAUSSIAN_COV_MIN)
    
    if i % int(NUM_GAUSSIANS_PER_CLASS/10) == 0:
        print(i/NUM_GAUSSIANS_PER_CLASS)

# Sample data
features = np.zeros((NUM_SAMPLES, DATA_DIM))
target = np.zeros((NUM_SAMPLES, 1))
for i in range(NUM_SAMPLES):
    class_index = np.random.randint(2)
    gaussian_index = np.random.randint(NUM_GAUSSIANS_PER_CLASS)
    if class_index == 0:
        features[i] = np.random.multivariate_normal(redMeans[gaussian_index], redCovs[gaussian_index])
    else:
        features[i] = np.random.multivariate_normal(blueMeans[gaussian_index], blueCovs[gaussian_index])
    
    target[i] = class_index
    
    if i % int(NUM_SAMPLES/10) == 0:
        print(i/NUM_SAMPLES)

# Set array types
features = features.astype('float32')
target = target.astype('int32')

# 75/25 train/test split
featuresTrain, featuresTest, targetTrain, targetTest = train_test_split(features, target)

np.save('data/xTrain.npy', featuresTrain)
np.save('data/xTest.npy', featuresTest)
np.save('data/yTrain.npy', targetTrain)
np.save('data/yTest.npy', targetTest)
