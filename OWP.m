function [ path ] = OWP( dtw )
% Return Optimal Warping Path based on Greedy algorithm
%   Detailed explanation goes here
[i,j] = size(dtw);
path = [];
%path = zeros(i+j,2);
%k=1;
% watch out for the condition, alter from the review paper
while (i>1) || (j>1)
    if (i == 1)
        j = j-1;
    elseif (j == 1)
        i = i-1;
    else
        choices = [dtw(i-1,j),dtw(i,j-1),dtw(i-1,j-1)];
        if dtw(i-1,j) == min(choices)
            i = i-1;
        elseif dtw(i,j-1) == min(choices)
            j = j-1;
        else
            i=i-1;
            j=j-1;
        end
        path = [path;i,j];
        % path(k,:) = [i,j];
        % k = k+1;
    end
        
end


end

