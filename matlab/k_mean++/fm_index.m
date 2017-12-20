function [ FM ] = fm_index( label1, label2 )
%fm_index computes the Fowlkes and Mallows index
%  label1 and label2 are two vectors containing the labels  of the data. 
% This index can be used to compare either two cluster label sets or 
% a cluster label  set with a true label set.

if numel(label1) ~= numel(label2)
    error('X and Y must have the same number of elements');
end

num_samples = numel(label1);
class_X = sort(unique(label1), 'ascend');
class_Y = sort(unique(label2), 'ascend');

R = numel(class_X);
C = numel(class_Y);

cont_table = zeros(R, C);

for i = 1:R
    for j = 1:C
        cont_table(i,j) = nnz(label1(:) == class_X(i) & label2(:) == class_Y(j));
    end
end

Z = sum(sum(cont_table.^2));
nR = sum(cont_table, 2);
nC = sum(cont_table);

bcR = zeros(R, 1);
for i = 1:R
    bcR(i) = nchoosek(nR(i), 2);
end

bcC = zeros(C, 1);
for j = 1:C
    bcC(j) = nchoosek(nC(j), 2);
end

FM = (1/2 * (Z - num_samples)) / sqrt(sum(bcR) * sum(bcC));

end

