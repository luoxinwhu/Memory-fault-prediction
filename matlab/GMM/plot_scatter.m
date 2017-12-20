function plot_scatter( X, L, K )
% PLOT_SCATTER 绘制二维、三维散点图
%  X    原始样本集 （matrix， n_samples * n_properties）
%  L    通过高斯混合聚类得到的类比标签集（vector, n_samples*1)
%  K     高斯混合模型中的簇类数目

[n_samples, n_properties] = size(X);

% 获得聚类类别数目
if isscalar(K)
    K_number = K;
else
    K_number = size(K, 1);
end

pr_color = zeros(n_samples, 3);
for k = 1: K_number
    color = [rand, rand, rand];            %建立颜色矩阵，随机给几个颜色
    csize=size(pr_color(L==k,:),1);        % 属于某一类的样本数目
    pr_color(L==k,:)=repmat(color,csize,1);    %对属于某一类下的点染色
end

figure(1);
scatter3(X(:, 1), X(:, 2), X(:, 3), 10, '*');
title('原始样本散点图（图中均为归一化值）');
xlabel('属性1');
ylabel('属性2');
zlabel('属性3');


figure(2);
if n_properties==2
    scatter(X(:, 1), X(:, 2), 10, pr_color);
elseif n_properties>=3
    scatter3(X(:, 1), X(:, 2), X(:, 3), 10, pr_color, 'filled');
end
title({['基于高斯混合聚类模型的样本散点图']; ['类别数目：', num2str(K), '  （图中均为归一化值）']});
xlabel('属性1');
ylabel('属性2');
zlabel('属性3');

end

