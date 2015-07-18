function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));  % 25 x 401
Theta2_grad = zeros(size(Theta2));  % 10 x 26
                      
a1 = [ ones(m,1) X];                % m x 401
z2 = a1*Theta1';                    % m x 25
a2 = [ ones(m,1) sigmoid(z2)];      % m x 26
z3 = a2*Theta2';                    % m x 10
a3 = sigmoid(z3);                   % m x 10
h = a3;                             % m x 10

% TODO Convert this into a matrix form
for k=1:num_labels
    yk = ( y == k );                % m x 1
    hk = h(:,k);                    % m x 1
    J = J -yk'*log(hk) - (1-yk')*log(1-hk);  % 1x1   
end

J = J/m;

% Add regularization
% We need to first remove the first column of each theta
t1 = Theta1(:,2:end);
t2 = Theta2(:,2:end);
unrolled = [t1(:) ; t2(:)];

J = J + lambda*unrolled'*unrolled/(2*m);

for t=1:m

    % We silce off the "t" th input element
    xt = X(t,:);                        % 1 x 400
    yt = y(t);                          % 1 x 1   has the value 1..10 

    % Step 1 - feed forward                     
    a1 = [ ones(1,1) xt];               % 1 x 401
    z2 = a1*Theta1';                    % 1 x 25
    a2 = [ ones(1,1) sigmoid(z2)];      % 1 x 26
    z3 = a2*Theta2';                    % 1 x 10
    a3 = sigmoid(z3);                   % 1 x 10
    h = a3;                             % 1 x 10

    % Step 2 - compute delta3
    yy = zeros(1,num_labels);
    yy(yt) = 1;                         % Set one of the index to 1 (to indicate the class)
    d3 = a3 - yy;                       % 1 x 10

    % Step 3 - compute delta2
    %temp = d3*Theta2;                 %  (1x10) (10x26) => 1x26
    %temp = temp(2:end);               % 1 x 25
    temp = d3*Theta2(:,2:end);         % (1x10) (10x25) => 1 x 25
    d2 = temp .* sigmoidGradient(z2);  % 1 x 25

    % Step 4
    Theta1_grad = Theta1_grad + d2'*a1; % (25x1)(1x401) => 25 x 401
    Theta2_grad = Theta2_grad + d3'*a2; % (10x1)(1x26) => 10x26
    
end

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
grad = grad/m;

% Add regularization
% Rememer to zero out the first column of both Theta1 and Theta2
t1 = Theta1;
t2 = Theta2;
t1(:,1) = 0;        % 25 x 401 where column 1 is all zeros
t2(:,1) = 0;        % 10 x 26  where column 1 is all zeroes
unrolled = [t1(:) ; t2(:)];
grad = grad + lambda*unrolled/m;

end
