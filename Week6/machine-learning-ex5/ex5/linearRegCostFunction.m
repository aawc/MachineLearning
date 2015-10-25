function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h_theta_x = X * theta;
theta_temp = theta;
theta_temp(1) = 0;

% == J ==
first_term = sum((h_theta_x - y) .^ 2) / (2*m);
second_term = lambda*sumsq(theta_temp)/(2*m);
J = first_term + second_term;

% == grad ==
first_term = sum((h_theta_x - y) .* X)' / m;
second_term = lambda * theta_temp / m;
grad = first_term + second_term;

% =========================================================================

grad = grad(:);

end
