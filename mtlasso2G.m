% Two-graph guided multi-task lasso by coordinate-descent algorithm
% Author: Xing Xu @ TTIC
% Last Update: 2012-4-25


function B_hat = mtlasso2G(X, Y, lambdas, G1, G2, tol, max_it)
% Input - X, observation matrix, size n by J
%         Y, label/task matrix, size n by K
%         lambdas, a vector of three regularization parameters
%         G1, information for the task graph
%         G2, information for the feature graph
%         tol, convergence criterion
%         max_it, maximum iteration allowed
% Output - B_hat, estimated coefficient matrix

K = size(Y, 2);   % K tasks
J = size(X, 2);   % J features

if nargin < 7, max_it = 200; end
if nargin < 6, tol = 1e-4 * J * K; end
if nargin < 5
    % Thresholds for graph
    corr_thres2 = 0.4;
    
    % Init Task Graph (G2) by correlations
    G2.C = tril(nanFilter(corrcoef(X)), -1);
    inds2 = find(abs(G2.C) > corr_thres2);
    G2.E = inds2subs(inds2, size(G2.C)) - 1;
    G2.C = G2.C(inds2);
    G2.W = abs(G2.C);
end
if nargin < 4
    % Thresholds for graph
    corr_thres1 = 0.4;
    
    % Init Task Graph (G1) by correlations
    G1.C = tril(nanFilter(corrcoef(Y)), -1);
    inds1 = find(abs(G1.C) > corr_thres1);
    G1.E = inds2subs(inds1, size(G1.C)) - 1;
    G1.C = G1.C(inds1);
    G1.W = abs(G1.C);
end

% Initial value
B = pinv(X) * Y;

% Coordinate Descent
tic
B_hat = mtlasso2G_CD(B, G1.W, G1.C, G1.E, G2.W, G2.C, G2.E, X, Y,...
    lambdas(1), lambdas(2), lambdas(3), tol, max_it);
toc