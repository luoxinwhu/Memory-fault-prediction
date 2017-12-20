function [ C ] = split_data( X, L, K )
%UNTITLED5 此处显示有关此函数的摘要
%   此处显示详细说明


% 根据类别标签将数据data分堆

for i =1: size(L,2)
    for j=1:K
        if L(i)==j
            C(j) = X(i);
        end
    end
end

            
            
%     if L(i)==1
%         C(1, j, :) = X(i,:);
%         j = j+1;
%     elseif L(i)==2
%         C(2, u, :) = X(i, :);
%         u = u+1;
%     elseif L(i)==3
%         C(3, p, :) =X(i, :);
%         p = p+1;
%     end
% end

end

