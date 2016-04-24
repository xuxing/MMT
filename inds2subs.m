% Transform matrix index from 1d to 2d
% Author: Xing Xu @ TTIC
% Last Update: 2012-4-25


function subs = inds2subs(inds, sizes)
% Input - inds, a vector of 1d indices to be processed
%       - sizes, the size of underlying matrix
% Output - subs, the same length with inds

num = length(inds);
subs = zeros(2, num);

for i = 1:num
    [sub1 sub2] = ind2sub(sizes, inds(i));
    subs(:, i) = [sub1; sub2];
end