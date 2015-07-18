function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% -- Daas Notes ---
%
%  m = number of training samples
%  n = number of features ( excluding the dummy column )
%  X = m x (n+1) Matrix
%      m = number of training samples
%      n = number of features. Add one for the dummy column
%  y = (mx1) vector
% theta = (n+1) x 1 vector

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

h = X*theta;   %  m x (n+1) dim * (n+1) x 1 dim
e = h - y;     %  m x 1 dim
J = e'*e/(2*m) %  1 x 1  

% =========================================================================

end
