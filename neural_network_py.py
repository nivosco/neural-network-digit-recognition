'''
July 2017
@author: Niv Vosco
'''

import math
import scipy.io
import numpy as np
from scipy.optimize import minimize

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def randInitializeWeights(in_layer_size, out_layer_size):
    return (np.random.rand(out_layer_size, in_layer_size + 1) * 2 * epsilon - epsilon)

def forwardPropogation(theta1, theta2):
    m,n = X.shape
    a1 = np.hstack((np.ones((X.shape[0], 1)), X))
    z2 = np.matmul(a1 ,theta1.T)
    a2 = np.hstack((np.ones((z2.shape[0], 1)), sigmoid(z2)))
    z3 = np.matmul(a2 ,theta2.T)
    a3 = sigmoid(z3)
    return a3

def costFunction(theta):
    cost = 0
    m,n = X.shape
    theta1 = (theta[:(hidden_layer_size * (input_layer_size + 1))]).reshape(hidden_layer_size, input_layer_size + 1)
    theta2 = (theta[(hidden_layer_size * (input_layer_size + 1)):]).reshape(num_labels, hidden_layer_size + 1)
    h = forwardPropogation(theta1, theta2)
    for i in range(0,m):
        for j in range(1, num_labels + 1):
            if (y.item(i,0) == j):
                cost = cost - math.log(h[i,j-1])
            else:
                cost = cost - math.log(1 - h[i,j-1])
    cost = cost / m
    T = np.sum(np.square(np.delete(theta1, -1, axis=1)))
    T = T + np.sum(np.square(np.delete(theta2, -1, axis=1)))
    cost = cost + ((T * lambda_reg) / (2 * m))
    return cost

def costFunctionGradient(theta):
    m,n = X.shape
    theta1 = (theta[:(hidden_layer_size * (input_layer_size + 1))]).reshape(hidden_layer_size, input_layer_size + 1)
    theta2 = (theta[(hidden_layer_size * (input_layer_size + 1)):]).reshape(num_labels, hidden_layer_size + 1)
    delta3 = np.zeros((m, num_labels))
    a1 = np.hstack((np.ones((X.shape[0], 1)), X))
    z2 = np.matmul(a1 ,theta1.T)
    a2 = np.hstack((np.ones((z2.shape[0], 1)), sigmoid(z2)))
    z3 = np.matmul(a2 ,theta2.T)
    a3 = sigmoid(z3)
    for i in range(0,m):
        for j in range(1, num_labels + 1):
            if (y.item(i,0) == j):
                delta3[i,j-1] = a3[i,j-1] - 1
            else:
                delta3[i,j-1] = a3[i,j-1]
    z2 = np.hstack((np.ones((z2.shape[0], 1)), z2))
    delta2 = np.multiply(np.matmul(delta3, theta2), (np.multiply(sigmoid(z2), (1 - sigmoid(z2)))))
    delta2 = (np.delete(delta2.T, 0, 0)).T
    theta1_grad = (np.matmul(a1.T, delta2) / m).T
    theta2_grad = (np.matmul(a2.T, delta3) / m).T
    theta1 = (np.delete(theta1.T, 0, 0)).T
    theta1 = np.hstack((np.zeros((theta1.shape[0], 1)), theta1))
    theta2 = (np.delete(theta2.T, 0, 0)).T
    theta2 = np.hstack((np.zeros((theta2.shape[0], 1)), theta2))
    theta1_grad = theta1_grad + (lambda_reg / m) * theta1
    theta2_grad = theta2_grad + (lambda_reg / m) * theta2
    theta_grad = np.append(theta1_grad.reshape(-1,1), theta2_grad.reshape(-1,1))
    return theta_grad

def gradientDescent(theta):
    m,n = X.shape
    a1 = np.hstack((np.ones((X.shape[0], 1)), X))
    delta3 = np.zeros((m, num_labels))
    for k in range(0, iter_grad):
        theta1 = (theta[:(hidden_layer_size * (input_layer_size + 1))]).reshape(hidden_layer_size, input_layer_size + 1)
        theta2 = (theta[(hidden_layer_size * (input_layer_size + 1)):]).reshape(num_labels, hidden_layer_size + 1)
        z2 = np.matmul(a1 ,theta1.T)
        a2 = np.hstack((np.ones((z2.shape[0], 1)), sigmoid(z2)))
        z3 = np.matmul(a2 ,theta2.T)
        a3 = sigmoid(z3)
        for i in range(0,m):
            for j in range(1, num_labels + 1):
                if (y.item(i,0) == j):
                    delta3[i,j-1] = a3[i,j-1] - 1
                else:
                    delta3[i,j-1] = a3[i,j-1]
        z2 = np.hstack((np.ones((z2.shape[0], 1)), z2))
        delta2 = np.multiply(np.matmul(delta3, theta2), (np.multiply(sigmoid(z2), (1 - sigmoid(z2)))))
        delta2 = (np.delete(delta2.T, 0, 0)).T
        theta1_grad = (np.matmul(a1.T, delta2) / m).T
        theta2_grad = (np.matmul(a2.T, delta3) / m).T
        theta1_t = (np.delete(theta1.T, 0, 0)).T
        theta1_t = np.hstack((np.zeros((theta1_t.shape[0], 1)), theta1_t))
        theta2_t = (np.delete(theta2.T, 0, 0)).T
        theta2_t = np.hstack((np.zeros((theta2_t.shape[0], 1)), theta2_t))
        theta1_grad = theta1_grad + (lambda_reg / m) * theta1_t
        theta2_grad = theta2_grad + (lambda_reg / m) * theta2_t
        theta_grad = (np.append(theta1_grad.reshape(-1,1), theta2_grad.reshape(-1,1))).reshape(-1,1)
        theta = theta - (alpha * theta_grad);
    return theta

def computeNumericalGradient(theta):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    for p in range(0,theta.shape[0]):
        perturb[p] = numerical_eps;
        loss1 = costFunction(theta - perturb)
        loss2 = costFunction(theta + perturb)
        numgrad[p] = (loss2 - loss1) / (2 * numerical_eps)
        perturb[p] = 0
    return numgrad

def predict(theta):
    m,n = X.shape
    pred = np.zeros((m, 1))
    theta1 = (theta[:(hidden_layer_size * (input_layer_size + 1))]).reshape(hidden_layer_size, input_layer_size + 1)
    theta2 = (theta[(hidden_layer_size * (input_layer_size + 1)):]).reshape(num_labels, hidden_layer_size + 1)
    h = forwardPropogation(theta1, theta2)
    for i in range(0,m):
        h_i = ((h[i,:]).T).reshape(-1,1)
        pred[i] = 1 + ((h_i.argmax(axis=0))[0])
    return pred

def calculateAccuracy(predictions):
    m,n = X.shape
    temp = predictions - y
    return (100 * ((np.count_nonzero(temp==0)) / m))

## Setup the parameters
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10 (10 represents 0)
epsilon = 0.12           # Range for parameter initialization
numerical_eps = 1e-4     # Numerical gradient parameter
lambda_reg = 0.5         # Regularization parameter
iter_grad = 1000         # Number of iterations to run gradient descent
alpha = 1                # Learning rate for gradient descent

## Load the data
data = scipy.io.loadmat('data.mat')  # trainig size of 5000 images
X = data["X"]                        # ndarray(5000,400)
y = data["y"]                        # ndarry(5000,1)

## Inialize the parameters
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)                                # parameters initialization for layer 1, ndarray(25,401)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)                                      # parameters initialization for layer 2, ndarray(10,26)
initial_nn_params = (np.append(initial_Theta1.reshape(-1,1), initial_Theta2.reshape(-1,1))).reshape(-1,1)  # Unroll parameters, ndarray(10285,1)

## Gradient checking
#numericalGradient = computeNumericalGradient(initial_nn_params)
#gradient = costFunctionGradient(initial_nn_params)
#diff = np.linalg.norm(numericalGradient-gradient)/np.linalg.norm(numericalGradient+gradient)
#print("Gradient checking value: ", diff)  # should be a small number ~ 1e-9

## Train the network
#res = minimize(costFunction, initial_nn_params, method='BFGS', jac=costFunctionGradient, options={'disp': True, 'maxiter': 50})
#nn_params = res.x
nn_params = gradientDescent(initial_nn_params)

## check accuaracy
pred = predict(nn_params)
print("\nTraining Set Accuracy: ", calculateAccuracy(pred))
