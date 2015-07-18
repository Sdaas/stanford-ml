function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

% mean computes the mean for each column in a matrix. Since
% columns in X are a feature, mean(x) will return the mean of
% each feature

m = size(X,1);          % number of rows. equals number of sample points
mu = mean(X);           % this is a 1 x (n+1) vector
mur = repmat(mu,m,1);   % stack the mean so that we have a m x (n+1) matrix

sigma = std(X);         % standard deviation of X
sr = repmat(sigma,m,1); % stack the sigma so that we have a m x (m+1) matrix

X_norm = ( X - mur) ./ sr;

% ============================================================

end
