function [ dtw , C ] = DTW( S1,S2 )
% Dynamic Time Warping Path to find the similarity of the two signals
%   Distance distribution, Dynamic Programming

n = length(S1);
m = length(S2);

dtw = zeros(n,m); % a table for DP
C = zeros(n,m); % pairwise distance cost matrix

% define C matrix
for i=1:n
    for k=1:m
        C(i,k) = abs(S1(i)-S2(k)); % absolute
    end
end

% Dynamic Programming to find the optimal path
% initialize the first row & column
for i=2:n
    dtw(i,1) = dtw(i-1,1) + C(i,1); 
end
for k=2:m
    dtw(1,k) = dtw(1,k-1) + C(1,k); 
end

% DP
for i=2:n
    for k=2:m
        choices = [dtw(i-1,k),dtw(i,k-1),dtw(i-1,k-1)];
        dtw(i,k) = C(i,k) + min(choices); % absolute
    end
end

end

