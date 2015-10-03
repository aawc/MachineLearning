function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

theta_t_x = X * theta;
h_theta_x = sigmoid(theta_t_x);

% === J ===
first_term = 0;
for row=1:m,
  first_term += (-y(row) * log(h_theta_x(row))) - (
    (1 - y(row)) * log(1 - h_theta_x(row)));
end

second_term = 0;
for col=2:n,
  second_term += theta(col)^2;
end
second_term *= (lambda/2);

J = (first_term + second_term) / m;
% === J ===

% === grad ===
for col=1:n,
  for row=1:m,
    % Only the first term for grad.
    grad(col) += (h_theta_x(row) - y(row)) * X(row, col);
  end
end
% For regularization in grad
lambda_vector = eye(n);
lambda_vector(1, 1) = 0;
lambda_vector *= lambda;
theta_lambda = lambda_vector * theta;
grad += theta_lambda;

grad /= m;
% === grad ===

% =============================================================

end
