%% Machine Learning Exercise - Digit recognition with Neural Network

%% Initialization
clear ; close all; clc

%% Setup the parameters
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % ("0" mapped to label 10)
epsilon = 0.12;           % Range for parameter initialization

%% Load the data
load('data.mat');

%% Inialize the parameters
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size, epsilon);   % parameters initialization for layer 1
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels, epsilon);         % parameters initialization for layer 2
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];                            % Unroll parameters

%% Train the neural network (back propagation)
options = optimset('MaxIter', 50);                                                                       % Run the optimization algorithm for 50 iterations
lambda = 0.5;                                                                                            % Use regularization parameter lambda equals 0.5
costFunction = @(p) costFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);      % Cost function to minimize
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);                                    % Run the optimization

%% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

%% check accuaracy (forward propagation)
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


