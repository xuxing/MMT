% Simulating, testing and comparing the performance of multi-task lassos
% Author: Xing Xu @ TTIC
% Last Update: 2011-9-19

clear;

% Simulation setups
N = 50;
K = 10;
J = 10;
diff = 10;
group_num = floor(sqrt(K * J / (K + J))) + 1;
sizes1 = randi(floor(sqrt(K)), 1, group_num);
sizes2 = randi(floor(sqrt(J)), 1, group_num);

% Generate synthetic data
tic
[X Y B] = simuData(N, K, J, sizes1, sizes2, diff);
fprintf('Data generation done...\n');
toc

% Normalization (Do not forget)
X = X - repmat(mean(X), N, 1);
Y = Y - repmat(mean(Y), N, 1);

% Run standard lasso
tic
lambdas_lasso = GDCV(@lasso, X, Y, 10, 3);
fprintf('1 Cross validation done...\n');
Bhat_la = lasso(X, Y, lambdas_lasso);
fprintf('1 Main program done...\n');
toc

% Run standard multi-task lasso
tic
lambdas_mt = GDCV(@mtlasso, X, Y, 10, 3);
fprintf('2 Cross Validation done...\n');
Bhat_mt = mtlasso(X, Y, lambdas_mt);
fprintf('2 Main program done...\n');
toc

% Run two-graph guided multi-task lasso
tic
lambdas_m2 = GDCV(@mtlasso2G, X, Y, [10 1 1], 3);
fprintf('3 Cross validation done...\n');
Bhat_m2 = mtlasso2G(X, Y, lambdas_m2);
fprintf('3 Main program done...\n');
toc

% Post-processing
thres = 1 * sqrt(log(J * K) / (N * K));
Bhat_la(abs(Bhat_la) < thres) = 0;
Bhat_mt(abs(Bhat_mt) < thres) = 0;
Bhat_m2(abs(Bhat_m2) < thres) = 0;

% Plot and visualize
figure;
subplot(2, 2, 1);
colorspy(B);
title('True B');
subplot(2, 2, 2);
colorspy(Bhat_la);
title(sprintf('lasso estimated(Error=%.3f)', sum(sum(abs(B - Bhat_la)))));
subplot(2, 2, 3);
colorspy(Bhat_mt);
title(sprintf('mtlasso estimated(Error=%.3f)', sum(sum(abs(B - Bhat_mt)))));
subplot(2, 2, 4);
colorspy(Bhat_m2);
title(sprintf('mtlasso2G estimated(Error=%.3f)', sum(sum(abs(B - Bhat_m2)))));