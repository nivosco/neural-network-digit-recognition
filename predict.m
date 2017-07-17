function p = predict(Theta1, Theta2, X)

m = size(X, 1);                  % size of the trainig set
num_labels = size(Theta2, 1);    % number of labels

%% forward propagation
h1 = sigmoid([ones(m, 1) X] * Theta1');    % add X0=1 and calculate the first layer output
h2 = sigmoid([ones(m, 1) h1] * Theta2');   % add X0=1 and calculate the second layer output
[dummy, p] = max(h2, [], 2);               % return the label with the maximum probability

end
