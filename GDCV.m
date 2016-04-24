% Gradient descent cross-validation to choose regularzation parameters
% Author: Xing Xu @ TTIC
% Last Update: 2012-4-25


function [lambdas history] = GDCV(F, X, Y, lambdas, k, tol, max_it)
% Input - F, function handler to evaluate, e.g. lasso, mtlasso2G
%         X, observation matrix, size n by J
%         Y, label matrix, size n by K
%         lambdas, initial values
%         k, number of folds for cross-validation
%         tol, allowed difference between two iterations, stop otherwise
%         max_it, maximum number of iterations allowed
% Output - lambdas, optimization result
%          history, trace histroy of parameters and errors

% Some (default) settings
if nargin < 7, max_it = 100; end
if nargin < 6, tol = 1e-1; end
if nargin < 5, k = 3;
stepsize = 1e-4;

% Gradient descent to optimize lambdas
flag_it = 0;
diff = tol + 1;
error_old = Inf;
history = [];
error0 = -1;

while flag_it < max_it && diff > tol
    deltaX = 1e-4;
    if error0 < 0
        error0 = CV(F, X, Y, lambdas, k);
        history = [history; [lambdas error0]];
    end
    
    % line search for each lambda
    n = length(lambdas);
    for i = 1:n
        % Do not continue if initial value is less than or equal to zero
        if lambdas(i) <= 0
            continue;
        end
        upper_bound = 1000;
        lower_bound = 0;
        
        % calculate gradient using finite differences
        lambdas_temp = lambdas; 
        lambda = lambdas(i); lambdas_temp(i) = lambda + deltaX;
        error1 = CV(F, X, Y, lambdas_temp, k);
        gradient = (error1 - error0) / deltaX;
        
        while upper_bound > lower_bound
            % Update the i-th position
            lambda_prime = lambda - upper_bound * stepsize * gradient;
            if lambda_prime < lambda / 10
                lambda_prime = lambda / 10;
                upper_bound = 0.9 * lambda / (stepsize * gradient);
            end
            if lambda_prime > 10 * lambda
                lambda_prime = 10 * lambda;
                upper_bound = 9 * lambda / abs(stepsize * gradient);
            end
            
            % Accept if error decreases
            lambdas_temp(i) = lambda_prime;
            error1_prime = CV(F, X, Y, lambdas_temp, k);
            if error1_prime > error0
                upper_bound = floor((upper_bound + lower_bound) / 2);
                continue;
            end
            error0 = error1_prime;
            lambdas = lambdas_temp;
            break;
        end
        
        history = [history; [lambdas error0]];
    end
    
    % update
    diff = error_old - error0; % Allow flucuation here
    error_old = error0;
    flag_it = flag_it + 1;
    if diff < error0 / 100
        break;
    end
end


function error = CV(F, X, Y, lambdas, k)
% Child function, return k-fold cross-validation error for a specific
% setting of parameters

N = size(X, 1);
block_size = floor(N / k);
error = 0;

for i = 1:k
    % Split data into training part and test part
    test_id = (i - 1) * block_size + 1 : min(i * block_size, N);
    training_id = setdiff(1:N, test_id);
    test_X = X(test_id, :);
    test_Y = Y(test_id, :);
    training_X = X(training_id, :);
    training_Y = Y(training_id, :);
    
    % run F and calc error
    B = feval(F, training_X, training_Y, lambdas);
    y_hat = test_X * B;
    error = error + sum(sum(abs(test_Y - y_hat)));
end
