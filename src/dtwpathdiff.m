function [ diff_cost ] = dtwpathdiff( P0,P1 )
% Input two signals' optimal dynamic time warping paths
%   Inputs are X and Y coordinates of different lengths, Mx2, Nx2

p0X = P0(:,1);
p0Y = P0(:,2);
p1X = P1(:,1);
p1Y = P1(:,2);

[dtw0] = DTW(p0X,p1X);
%[P0] = OWP(dtw0);
disp(dtw0(end))

[dtw1] = DTW(p0Y,p1Y);
disp(dtw1(end))
diff_cost = dtw0(end) + dtw1(end);

end

