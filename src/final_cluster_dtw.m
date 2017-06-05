% process clustering based on dynamic time warping path
clear;
load('final_data')
% item, star, year, month, day, week
num_item = max(item)+1;
count_mat = zeros(6,12,num_item); % 2010~2015, 12 months, #item
data_mat = zeros(6,12,num_item); % 2010~2015, 12 months, #item
num_mat = zeros(6,12,num_item);

for i=1:length(item)
    if (year(i)>2009)
        data_mat(year(i)-2009,month(i),item(i)+1) = data_mat(year(i)-2009,month(i),item(i)+1) + star(i);
        count_mat(year(i)-2009,month(i),item(i)+1) = count_mat(year(i)-2009,month(i),item(i)+1) + 1;
    end
end

% take average rate
for i=1:num_item
    for k=1:6
       for m=1:12
          if count_mat(k,m,i)>0 
            data_mat(k,m,i) = data_mat(k,m,i)/count_mat(k,m,i);
            num_mat(k,m,i) = 1;
          end
       end
    end
end

count_item = zeros(1,num_item);
for i=1:num_item
   count_item(i) = sum(sum(count_mat(:,:,i))); 
end

[max_val,max_idx] = max(count_item);

% to do DTW
proc_mat = zeros(num_item,12);
for i=1:num_item
    %proc_mat(i,:) = sum(data_mat(:,:,i),1)./sum(count_mat(:,:,i))-sum(sum(data_mat(:,:,i),1))/72;
    proc_mat(i,:) = sum(data_mat(:,:,i),1)./sum(num_mat(:,:,i))-sum(sum(data_mat(:,:,i)))/sum(sum(num_mat(:,:,i),1));
end

item_map = 1:1:num_item;
num_cluster = 20;
C = cell(num_cluster,1);

% remove all zero data
Zrm = find(count_item==0);
item_map(Zrm) = [];
proc_mat(Zrm,:) = [];
count_item(Zrm) = [];

for k=1:num_cluster
% DTW cost matrix (distance matrix)
%
D1 = zeros(1,length(proc_mat));
parfor i=1:length(proc_mat)
    [dtw0] = DTW(proc_mat(i,:),proc_mat(max_idx,:));
       [mP0] = OWP(dtw0);
      [dtw1] = magDTW(proc_mat(i,:),proc_mat(max_idx,:));
      [mP1] = OWP(dtw1);
      D1(i) = dtwpathdiff(mP0,mP1);
end
C1 = find(D1==0);
if numel(C1) == 0
    k = k-1;
    max_idx = randi(length(proc_mat),1,1);
else
    C{k} = item_map(C1);
    proc_mat(C1,:) = [];
    item_map(C1) = [];
    count_item(C1) = [];
    [max_val,max_idx] = max(count_item);
end

end