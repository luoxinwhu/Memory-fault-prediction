%% ��ջ�������
clc;
clear;
close all;

%% ��������
X = load('datasets/iris.txt');
[n_samples, n_properties] = size(X);

% %% ���ݹ�һ��Ԥ����
[X_scale, ~] = mapminmax(X', 0, 1);   % MATLAB�Դ��Ĺ�һ����������Ĭ���й�һ����������Ҫ��wine ÿһ���������ν��й�һ���������ת��
X = X_scale' ;  % wine_scale��n_properties, n_samples��

%% ����ִ�и�˹��Ͼ���
K = 4;
[Px, GMM_model] = gmm(X, K);
%display(GMM_model);
[~, L] = max(Px, [], 2);   %ѡ����������Ǹ�������������ǩ��L


%% ���չʾ
% ɢ��ͼ
plot_scatter(X, L, K);

% ����ɢ��ͼ
plot_heatMap_scatter(X, L, K);





