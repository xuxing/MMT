% Visualize matrix patterns in an image
% Author: Xing Xu @ TTIC
% Last Update: 2012-4-25


function colorspy(M)
% Input - M, the matrix to visualize

% Transform values of M to image scales
M = -M;
M = sign(M) .* abs(M) .^ .7;
M = M / max(max(abs(M))); % Normalize
[m n] = size(M);

% To color map format and then generate image
CmapM = zeros(m,n,3);
for i = 1:m
    for j = 1:n
        CmapM(i, j, :) = [1 - max(0, -M(i, j)), 1 - abs(M(i, j)),...
            1 - max(0, M(i, j))];
    end
end
image(CmapM);