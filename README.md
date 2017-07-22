# neural-network-digit-recognition
This is a neural network matlab/python implementation for digit recognition

Matlab implementation: 
 - main.m - loads the data, train the network and check the accuracy of the model
 - randInitializeWeights.m - initializes the parameters
 - costFunction.m - calculates the cost
 - sigmoid.m - calculate the sigmoid function
 - predict.m - calculates the predictions of the model
 
Python implementation: 
 - neural_network_py.py - loads the data, initialize the parameters, train the network and check the accuracy of the model

Neural network features:
 - The input to the network is 20x20 image (5000 sampels in data.mat)
 - The network has one hidden layer with 25 hidden units (configurable)
 - The output is a 10x1 probability vector for eah digit ("0" is mapped to "10")
 - The obtained accuracy for the training set is about 95%
