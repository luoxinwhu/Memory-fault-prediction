%% 清空环境变量
close all;
clear;
clc;
format compact;

%% 加载数据
load('iris.txt');

%% 数据归一化预处理
[X_scale, ~] = mapminmax(X', 0, 1);   % MATLAB自带的归一化处理函数（默认行归一化），而需要对wine 每一属性列依次进行归一化，因此先转置
X_scale = X_scale' ;  % wine_scale【n_properties, n_samples】

%% k-means++ algorithm
K = 3;  % 簇的数量
[L, U] = kmeans(X_scale', K); % L【1*n_samples】是标签集，C【n_properties*K】是每一类的中心样本点


%% 聚类性能度量
% Silhouette Coefficient 轮廓系数，用于确定最佳k值
rng('default');
silhouette(X_scale, L);

% 外部指标
% Jaccard_array = pdist2(L_ref, L, 'jaccard');
% JC = mean(Jaccard_array(:))
% FMI = fm_index(L_ref, L)
%[AR, RI, MI, HI] = rand_index(L_ref, L)

% 内部指标
% pdist(A)获得矩阵A中各对行向量之间的相互距离，squareform()将一维矩阵变换为对应的M*N阶形式
distM = squareform(pdist(X_scale)); %差异度矩阵
DI = dunn_index(K, distM, L)


%% PLOT AREA 
%plotbox_data_prop(X, categories, '原始数据属性总览');
plot_samples_center(X_scale, U, L, K);       % 样本点&中心点空间分布





