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

% compute cost -without- the regularization term
h = X*theta;   %  m x (n+1) dim * (n+1) x 1 dim
e = h - y;     %  m x 1 
J = e'*e/(2*m); %  1 x 1  

% add the regularization term to the cost
t = theta;
t(1) = 0;
J = J + lambda/(2*m)*t'*t; 



% =========================================================================

grad = grad(:);

end
