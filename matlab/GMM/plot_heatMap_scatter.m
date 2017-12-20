function plot_heatMap_scatter( X, L, K )
%UNTITLED3 此处显示有关此函数的摘要
%   X 原始数据集

gm = fitgmdist(X, K);
P = posterior(gm, X);

figure;
symb = ['*', 'o', 's', '<'];
for k = 1:K
    cidx = find(L==k);
    scatter(X(cidx, 1), X(cidx, 2), 20, P(cidx,1), symb(k), 'fill');
    hold on;
end
clrmap = jet(80);
colormap(clrmap(9:72, :));
ylabel(colorbar, '类别1的后验概率');
title({['基于高斯混合聚类模型的样本点热力图']; ['类别数目：', num2str(K), '  （图中均为归一化值）']});
grid on;

end

