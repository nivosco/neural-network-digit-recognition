'''
July 2017
@author: Niv Vosco
'''

import math
import scipy.io
import numpy as np

#from scipy.optimize import minimize
# def gradientDescent2(theta):
    # m,n = X.shape
    # a1 = np.hstack((np.ones((X.shape[0], 1)), X))
    # delta3 = np.zeros((m, num_labels))
    # print("Start: gradient descent")
    # for k in range(0, iter_grad):
        # if (k % (iter_grad / 10) == 0):
            # print("Iteration %d" % k)
        # theta1 = (theta[:(hidden_layer_size * (input_layer_size + 1))]).reshape(hidden_layer_size, input_layer_size + 1)
        # theta2 = (theta[(hidden_layer_size * (input_layer_size + 1)):]).reshape(num_labels, hidden_layer_size + 1)
        # z2 = np.matmul(a1 ,theta1.T)
        # a2 = np.hstack((np.ones((z2.shape[0], 1)), sigmoid(z2)))
        # z3 = np.matmul(a2 ,theta2.T)
        # a3 = sigmoid(z3)
        # for i in range(0,m):
            # for j in range(1, num_labels + 1):
                # if (y.item(i,0) == j):
                    # delta3[i,j-1] = a3[i,j-1] - 1
                # else:
                    # delta3[i,j-1] = a3[i,j-1]
        # z2 = np.hstack((np.ones((z2.shape[0], 1)), z2))
        # delta2 = np.multiply(np.matmul(delta3, theta2), (np.multiply(sigmoid(z2), (1 - sigmoid(z2)))))
        # delta2 = (np.delete(delta2.T, 0, 0)).T
        # theta1_grad = (np.matmul(a1.T, delta2) / m).T
        # theta2_grad = (np.matmul(a2.T, delta3) / m).T
        # theta1_t = (np.delete(theta1.T, 0, 0)).T
        # theta1_t = np.hstack((np.zeros((theta1_t.shape[0], 1)), theta1_t))
        # theta2_t = (np.delete(theta2.T, 0, 0)).T
        # theta2_t = np.hstack((np.zeros((theta2_t.shape[0], 1)), theta2_t))
        # theta1_grad = theta1_grad + (lambda_reg / m) * theta1_t
        # theta2_grad = theta2_grad + (lambda_reg / m) * theta2_t
        # theta_grad = (np.append(theta1_grad.reshape(-1,1), theta2_grad.reshape(-1,1))).reshape(-1,1)
        # theta = theta - (alpha * theta_grad)
    # print("Iteration %d" % iter_grad)
    # print("Done: gradient descent")
    # return theta

# def gradientDescent3(theta):
    # m,n = X.shape
    # a1 = np.hstack((np.ones((X.shape[0], 1)), X))
    # delta4 = np.zeros((m, num_labels))
    # print("Start: gradient descent")
    # for k in range(0, iter_grad):
        # ptr = 0
        # if (k % (iter_grad / 10) == 0):
            # print("Iteration %d" % k)
        # theta1 = (theta[ptr:(ptr+(hidden_layer_size * (input_layer_size + 1)))]).reshape(hidden_layer_size, input_layer_size + 1)
        # ptr += (hidden_layer_size * (input_layer_size + 1))
        # theta2 = (theta[ptr:(ptr+(hidden_layer_size * (hidden_layer_size + 1)))]).reshape(hidden_layer_size, hidden_layer_size + 1)
        # ptr += (hidden_layer_size * (hidden_layer_size + 1))
        # theta3 = (theta[ptr:(ptr+(hidden_layer_size * (input_layer_size + 1)))]).reshape(num_labels, hidden_layer_size + 1)
        # z2 = np.matmul(a1 ,theta1.T)
        # a2 = np.hstack((np.ones((z2.shape[0], 1)), sigmoid(z2)))
        # z3 = np.matmul(a2 ,theta2.T)
        # a3 = np.hstack((np.ones((z3.shape[0], 1)), sigmoid(z3)))
        # z4 = np.matmul(a3 ,theta3.T)
        # a4 = sigmoid(z4)
        # for i in range(0,m):
            # for j in range(1, num_labels + 1):
                # if (y.item(i,0) == j):
                    # delta4[i,j-1] = a4[i,j-1] - 1
                # else:
                    # delta4[i,j-1] = a4[i,j-1]

        # theta3_grad = (np.matmul(a3.T, delta4) / m).T
        # theta3_t = (np.delete(theta3.T, 0, 0)).T
        # theta3_t = np.hstack((np.zeros((theta3_t.shape[0], 1)), theta3_t))
        # theta3_grad = theta3_grad + (lambda_reg / m) * theta3_t

        # z3 = np.hstack((np.ones((z3.shape[0], 1)), z3))
        # delta3 = np.multiply(np.matmul(delta4, theta3), (np.multiply(sigmoid(z3), (1 - sigmoid(z3)))))
        # delta3 = (np.delete(delta3.T, 0, 0)).T
        # theta2_grad = (np.matmul(a2.T, delta3) / m).T
        # theta2_t = (np.delete(theta2.T, 0, 0)).T
        # theta2_t = np.hstack((np.zeros((theta2_t.shape[0], 1)), theta2_t))
        # theta2_grad = theta2_grad + (lambda_reg / m) * theta2_t

        # z2 = np.hstack((np.ones((z2.shape[0], 1)), z2))
        # delta2 = np.multiply(np.matmul(delta3, theta2), (np.multiply(sigmoid(z2), (1 - sigmoid(z2)))))
        # delta2 = (np.delete(delta2.T, 0, 0)).T
        # theta1_grad = (np.matmul(a1.T, delta2) / m).T
        # theta1_t = (np.delete(theta1.T, 0, 0)).T
        # theta1_t = np.hstack((np.zeros((theta1_t.shape[0], 1)), theta1_t))
        # theta1_grad = theta1_grad + (lambda_reg / m) * theta1_t
        # theta_grad = (np.append(theta1_grad.reshape(-1,1), theta2_grad.reshape(-1,1))).reshape(-1,1)
        # theta_grad = (np.append(theta_grad, theta3_grad.reshape(-1,1))).reshape(-1,1)
        # theta = theta - (alpha * theta_grad)
    # print("Iteration %d" % iter_grad)
    # print("Done: gradient descent")
    # return theta

# def costFunction(theta):
    # cost = 0
    # m,n = X.shape
    # theta1 = (theta[:(hidden_layer_size * (input_layer_size + 1))]).reshape(hidden_layer_size, input_layer_size + 1)
    # theta2 = (theta[(hidden_layer_size * (input_layer_size + 1)):]).reshape(num_labels, hidden_layer_size + 1)
    # h = forwardPropogation(theta1, theta2)
    # for i in range(0,m):
        # for j in range(1, num_labels + 1):
            # if (y.item(i,0) == j):
                # cost = cost - math.log(h[i,j-1])
            # else:
                # cost = cost - math.log(1 - h[i,j-1])
    # cost = cost / m
    # T = np.sum(np.square(np.delete(theta1, -1, axis=1)))
    # T = T + np.sum(np.square(np.delete(theta2, -1, axis=1)))
    # cost = cost + ((T * lambda_reg) / (2 * m))
    # return cost

# def costFunctionGradient(theta):
    # m,n = X.shape
    # theta1 = (theta[:(hidden_layer_size * (input_layer_size + 1))]).reshape(hidden_layer_size, input_layer_size + 1)
    # theta2 = (theta[(hidden_layer_size * (input_layer_size + 1)):]).reshape(num_labels, hidden_layer_size + 1)
    # delta3 = np.zeros((m, num_labels))
    # a1 = np.hstack((np.ones((X.shape[0], 1)), X))
    # z2 = np.matmul(a1 ,theta1.T)
    # a2 = np.hstack((np.ones((z2.shape[0], 1)), sigmoid(z2)))
    # z3 = np.matmul(a2 ,theta2.T)
    # a3 = sigmoid(z3)
    # for i in range(0,m):
        # for j in range(1, num_labels + 1):
            # if (y.item(i,0) == j):
                # delta3[i,j-1] = a3[i,j-1] - 1
            # else:
                # delta3[i,j-1] = a3[i,j-1]
    # z2 = np.hstack((np.ones((z2.shape[0], 1)), z2))
    # delta2 = np.multiply(np.matmul(delta3, theta2), (np.multiply(sigmoid(z2), (1 - sigmoid(z2)))))
    # delta2 = (np.delete(delta2.T, 0, 0)).T
    # theta1_grad = (np.matmul(a1.T, delta2) / m).T
    # theta2_grad = (np.matmul(a2.T, delta3) / m).T
    # theta1 = (np.delete(theta1.T, 0, 0)).T
    # theta1 = np.hstack((np.zeros((theta1.shape[0], 1)), theta1))
    # theta2 = (np.delete(theta2.T, 0, 0)).T
    # theta2 = np.hstack((np.zeros((theta2.shape[0], 1)), theta2))
    # theta1_grad = theta1_grad + (lambda_reg / m) * theta1
    # theta2_grad = theta2_grad + (lambda_reg / m) * theta2
    # theta_grad = np.append(theta1_grad.reshape(-1,1), theta2_grad.reshape(-1,1))
    # return theta_grad

# def computeNumericalGradient(theta):
    # numgrad = np.zeros(theta.shape)
    # perturb = np.zeros(theta.shape)
    # for p in range(0,theta.shape[0]):
        # perturb[p] = numerical_eps;
        # loss1 = costFunction(theta - perturb)
        # loss2 = costFunction(theta + perturb)
        # numgrad[p] = (loss2 - loss1) / (2 * numerical_eps)
        # perturb[p] = 0
    # return numgrad

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def randInitializeWeights(in_layer_size, out_layer_size):
    return (np.random.rand(out_layer_size, in_layer_size + 1) * 2 * epsilon - epsilon)

def forwardPropogation(theta):
    m,n = X.shape
    ptr_first = 0
    ptr_last = (hidden_layer_size * (input_layer_size + 1))
    z = X
    a = np.hstack((np.ones((z.shape[0], 1)), z))
    for i in range(0,num_hidden_layers+1):
        thetal = (theta[ptr_first:ptr_last]).reshape(layer_size[i+1], layer_size[i] + 1)
        ptr_first = ptr_last
        if (i < num_hidden_layers):
            ptr_last += (layer_size[i+2] * (layer_size[i+1] + 1))
        z = np.matmul(a ,thetal.T)
        a = np.hstack((np.ones((z.shape[0], 1)), sigmoid(z)))
    return a[:,1:]

def getParametersOfLayer(theta, layer):
    m,n = X.shape
    ptr_first = 0
    ptr_last = (hidden_layer_size * (input_layer_size + 1))
    a = np.hstack((np.ones((X.shape[0], 1)), X))
    theta1 = (theta[ptr_first:ptr_last]).reshape(hidden_layer_size, input_layer_size + 1)
    z = np.matmul(a ,theta1.T)
    for i in range(0,layer):
        ptr_first = ptr_last
        if (i < num_hidden_layers):
            ptr_last += (layer_size[i+2] * (layer_size[i+1] + 1))
        thetal = (theta[ptr_first:ptr_last]).reshape(layer_size[i+2], layer_size[i+1] + 1)
        a = np.hstack((np.ones((z.shape[0], 1)), sigmoid(z)))
        z = np.matmul(a ,thetal.T)
    return [a,z]

def gradientDescent(theta):
    m,n = X.shape
    print("Start: gradient descent")
    for k in range(0, iter_grad):
        if (k % (iter_grad / 10) == 0):
            print("Iteration %d" % k)
        deltal = np.zeros((m, num_labels))
        h = forwardPropogation(theta)
        for i in range(0,m):
            for j in range(1, num_labels + 1):
                if (y.item(i,0) == j):
                    deltal[i,j-1] = h[i,j-1] - 1
                else:
                    deltal[i,j-1] = h[i,j-1]
        [a,z] = getParametersOfLayer(theta, num_hidden_layers)
        theta_grad = (np.matmul(a.T, deltal) / m).T
        theta_grad_t = (np.delete(theta_grad.T, 0, 0)).T
        theta_grad_t = np.hstack((np.zeros((theta_grad_t.shape[0], 1)), theta_grad_t))
        theta_grad = theta_grad + (lambda_reg / m) * theta_grad_t
        ptr_last = theta.shape[0]
        ptr_first = 0
        for i in range(0,num_hidden_layers):
            ptr_first = ptr_first + (layer_size[i+1] * (layer_size[i] + 1))
        for i in range(1,num_hidden_layers+1):
            thetal = (theta[ptr_first:ptr_last]).reshape(layer_size[num_hidden_layers + 2 - i], layer_size[num_hidden_layers + 1 - i] + 1)
            ptr_last = ptr_first
            ptr_first = 0
            for j in range(0,num_hidden_layers-i):
                ptr_first = ptr_first + (layer_size[j+1] * (layer_size[j] + 1))
            [a,z] = getParametersOfLayer(theta, num_hidden_layers - i)
            z = np.hstack((np.ones((z.shape[0], 1)), z))
            delta = np.multiply(np.matmul(deltal, thetal), (np.multiply(sigmoid(z), (1 - sigmoid(z)))))
            delta = (np.delete(delta.T, 0, 0)).T
            theta_grad_t = (np.matmul(a.T, delta) / m).T
            theta_grad_tt = (np.delete(theta_grad_t.T, 0, 0)).T
            theta_grad_tt = np.hstack((np.zeros((theta_grad_tt.shape[0], 1)), theta_grad_tt))
            theta_grad_t = theta_grad_t + (lambda_reg / m) * theta_grad_tt
            theta_grad = (np.append(theta_grad_t.reshape(-1,1), theta_grad.reshape(-1,1))).reshape(-1,1)
            deltal = delta
        theta = theta - (alpha * theta_grad)
    print("Iteration %d" % iter_grad)
    print("Done: gradient descent")
    return theta

def predict(theta):
    m,n = X.shape
    pred = np.zeros((m, 1))
    h = forwardPropogation(theta)
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
num_hidden_layers = 2    # 2 hidden layers
num_labels = 10          # 10 labels, from 1 to 10 (10 represents 0)
epsilon = 0.12           # Range for parameter initialization
numerical_eps = 1e-4     # Numerical gradient parameter
lambda_reg = 0.5         # Regularization parameter
iter_grad = 1000         # Number of iterations to run gradient descent
alpha = 1                # Learning rate for gradient descent
layer_size = [0] * (num_hidden_layers + 2)
layer_size[0] = input_layer_size
layer_size[num_hidden_layers + 1] = num_labels
for i in range(1,num_hidden_layers+1):
    layer_size[i] = hidden_layer_size

## Load the data
data = scipy.io.loadmat('data.mat')  # trainig size of 5000 images
X = data["X"]                        # ndarray(5000,400)
y = data["y"]                        # ndarry(5000,1)

## Inialize the parameters
initial_nn_params = np.zeros((0,0))
for i in range(0,num_hidden_layers + 1):
    initial_theta = randInitializeWeights(layer_size[i], layer_size[i+1])                                   # parameters initialization for layer 1, ndarray(25,401)
    initial_nn_params = (np.append(initial_nn_params, initial_theta.reshape(-1,1))).reshape(-1,1)           # parameters initialization for layer 2, ndarray(10,26)

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
