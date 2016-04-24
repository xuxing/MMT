% filter NaN values in a matrix by setting them to zero
% Author: Xing Xu @ TTIC
% Last Update: 2012-4-25


function C = nanFilter(C)

C(isnan(C)) = 0;