%% 清空环境变量
clc;
clear;
close all;

%% 加载数据
X = load('datasets/iris.txt');
[n_samples, n_properties] = size(X);

% %% 数据归一化预处理
[X_scale, ~] = mapminmax(X', 0, 1);   % MATLAB自带的归一化处理函数（默认行归一化），而需要对wine 每一属性列依次进行归一化，因此先转置
X = X_scale' ;  % wine_scale【n_properties, n_samples】

%% 迭代执行高斯混合聚类
K = 4;
[Px, GMM_model] = gmm(X, K);
%display(GMM_model);
[~, L] = max(Px, [], 2);   %选择概率最大的那个数，输出聚类标签集L


%% 结果展示
% 散点图
plot_scatter(X, L, K);

% 热力散点图
plot_heatMap_scatter(X, L, K);





