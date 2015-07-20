function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

tol = 1e-3;         % input to svmTrain
max_passes = 5;     % input to svmTrain

Clist = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
sigmaList = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 ]';

%Clist = [30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01 ];
%sigmaList = [30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01 ];

emin = length(yval)*(10);
E = zeros(length(Clist),length(sigmaList));  % For debugging

for i=1:length(Clist)
    c = Clist(i);
    for j=1:length(sigmaList)
        s = sigmaList(j);

        % create a model
        model= svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s), tol, max_passes); 

        % and use it to predict against the cross-validation set
        predictions =  svmPredict(model, Xval);

        % and compute the prediction error
        e = mean(double(predictions ~= yval));

        % for debugging
        E(i,j) = e;

        % Check if this is a minimum
        if ( e < emin )
            C = c;
            sigma = s;
            emin = e;
        endif

        fprintf('Current C = %f sigma = %f e = %f \n', c,s,e);
        fprintf('   Best C = %f sigma = %f e = %f \n', C,sigma,emin);

    end
end

E

% =========================================================================

end
