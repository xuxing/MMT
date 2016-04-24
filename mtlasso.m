% Standard multi-task lasso by coordinate-descent algorithm
% Author: Xing Xu @ TTIC
% Last Update: 2012-4-25


function B = mtlasso(X, Y, lambda, tol, max_it)
% Input - X, feature matrix
%         Y, label/task matrix
%         lambda, ell_1/ell_2 regularization parameter
%         tol, convergence criterion
%         max_it, maximum iteration allowed
% Output - B, learned coefficient matrix

K = size(Y, 2);   % K tasks
J = size(X, 2);   % J features

if nargin < 5, max_it = 1e2; end
if nargin < 4, tol = 1e-4 * J * K; end

% Initialization
B = pinv(X) * Y;
D = (sum(abs(B')) / sum(sum(abs(B))))';

flag_it = 0;    % Current number of iterations
diff = tol + 1;
% Coordinate-descent loop
while diff > tol && flag_it < max_it,
    B_old = B;

    % C implementation of updating parameters
    [B, D] = grouplasso_CD( B, X, Y, D, lambda );
    
    % Calculate improvement between successive iterations
    diff = sum(sum(abs(B - B_old)));
    flag_it = flag_it + 1;
end