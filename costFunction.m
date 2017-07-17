function [J grad] = costFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

% Reshape nn_params back into the parameters Theta1 and Theta2
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Size of the training set
m = size(X, 1);

% forward propagation
a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
h = a3;
for i=1:m
    for j=1:num_labels
        if (y(i) == j) 
            J = J - log(h(i,j));
        else
            J = J - log(1 - h(i,j));
        end
    end
end
J = J / m;
T = sum(sum((Theta1(:,2:(input_layer_size+1)) .^ 2)));
T = T + sum(sum((Theta2(:,2:(hidden_layer_size+1)) .^ 2)));
J = J + ((T*lambda) / (2*m));

% backpropagation
delta3 = zeros(m, num_labels);
for i=1:m
    for j=1:num_labels
        if (y(i) == j) 
            delta3(i,j) = a3(i,j) - 1;
        else
            delta3(i,j) = a3(i,j);
        end
    end
end
z2 = [ones(size(z2, 1), 1) z2];
delta2 = delta3 * Theta2 .* (sigmoid(z2) .* (1 - sigmoid(z2)));
delta2 = delta2(:,2:end);
Theta1_grad = ((a1' * delta2) ./ m)';
Theta2_grad = ((a2' * delta3) ./ m)';
Theta1 = Theta1(:,2:end);
Theta1 = [zeros(size(Theta1, 1), 1) Theta1];
Theta2 = Theta2(:,2:end);
Theta2 = [zeros(size(Theta2, 1), 1) Theta2];
Theta1_grad = Theta1_grad + (lambda / m) * Theta1;
Theta2_grad = Theta2_grad + (lambda / m) * Theta2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
