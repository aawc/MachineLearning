function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
n = size(theta);
grad = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

for row=1:m,
  xi = X(row, :);
  theta_t_x = xi * theta;
  h_theta_x(row) = sigmoid(theta_t_x);
  expression = (-y(row) * log(h_theta_x(row))) - (
    (1 - y(row)) * log(1 - h_theta_x(row)));
  J += expression;
end

for col=1:n,
  for row=1:m,
    expression = (h_theta_x(row) - y(row)) * X(row, col);
    grad(col) += expression;
  end
  grad(col) /= m;
end

J = J / m;

% =============================================================

end
