function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Do the first layer. Input is X. Weight is Theta1. Output is a1

% Add an extra input x0 = 1
% Add ones to the X data matrix
X = [ones(m, 1) X];                     % x is  m x 401            
a1 = sigmoid(X*Theta1');                % a1 is m x 25

% Add an extra input a1_0 = 1
a1 = [ones(m,1) a1];                    % a1 is now m x 26
a2 = sigmoid(a1*Theta2');               % a2 is now m x 10

[xxx,p]=max(a2,[],2);          % find the index that has the maximum value ... 
                              % that is the predicted class

% =========================================================================


end
