% Interfacial function for CD optimized standard lasso
% Can do multiple (K > 1) tasks of lassos at the same time
% Author: Xing Xu @ TTIC
% Last Update: 2012-4-25


function B_hat = lasso(X, Y, lambda, tol, max_it)
% Input - X, observation matrix, size n by J
%         Y, label matrix, size n by K
%         lambda, ell_1 regularization parameter
%         tol, convergence criterion
%         max_it, maximum iteration allowed
% Output - B_hat, estimated coefficient matrix

J = size(X, 2);
K = size(Y, 2);

if nargin < 4, tol = 1e-4 * J * K; end
if nargin < 5, max_it = 200; end

% Initialize
B = pinv(X) * Y;

% Coordinate Descent
tic
B_hat = mtlasso2G_CD(B, [], [], [], [], [], [], X, Y,...
    lambda, 0, 0, tol, max_it);
toc