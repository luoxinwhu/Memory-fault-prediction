function plot_samples_center( X, U, L, K)
% plot_samples 绘制聚类过程中样本点和中心点分布情况
% X 原始样本集 【n_samples * n_properties】
% U 通过k-means++ 算法得到的中心点集 【K * n_properties】
% L 通过k-means++ 算法得到的类别标签集 【n_samples * 1】
% K 聚类的数目


        
P1 = figure;
symb = {'b*', 'g*', 'y*', 'm*', 'c*', 'k*'};
MarkFace = {[1,0,0], [0.8, 0, 1], [1, 0.5, 0], [0.8, 0, 0], [0.4, 1, 0], [0, 0.9, 0.5]};
for k=1:K
    clust = find(L==k);
    plot(X(clust, 3), X(clust, 5), symb{k});    % 选取第3、5个属性绘制样本散点图
    legend_str{k}=['类别',num2str(k)];   %定义每一个legend文本
    hold on;
end
plot(U(:, 3), U(:, 5), 'sr', 'MarkerSize', 10, 'MarkerFace', MarkFace{k});   %绘制中心点

% 设定图像样式
title('样本点和中心点散点图');
xlabel('属性3 Excess kurtosis of the integrated profile');
ylabel('属性5 Mean of the DM-SNR curve');
xlim([-0.2, 1.2]);
ylim([-0.2, 1.2]);
legend(legend_str ,'中心点'); %统一添加图例
end

