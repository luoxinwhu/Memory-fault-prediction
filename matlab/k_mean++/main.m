%% ��ջ�������
close all;
clear;
clc;
format compact;

%% ��������
load('iris.txt');

%% ���ݹ�һ��Ԥ����
[X_scale, ~] = mapminmax(X', 0, 1);   % MATLAB�Դ��Ĺ�һ����������Ĭ���й�һ����������Ҫ��wine ÿһ���������ν��й�һ���������ת��
X_scale = X_scale' ;  % wine_scale��n_properties, n_samples��

%% k-means++ algorithm
K = 3;  % �ص�����
[L, U] = kmeans(X_scale', K); % L��1*n_samples���Ǳ�ǩ����C��n_properties*K����ÿһ�������������


%% �������ܶ���
% Silhouette Coefficient ����ϵ��������ȷ�����kֵ
rng('default');
silhouette(X_scale, L);

% �ⲿָ��
% Jaccard_array = pdist2(L_ref, L, 'jaccard');
% JC = mean(Jaccard_array(:))
% FMI = fm_index(L_ref, L)
%[AR, RI, MI, HI] = rand_index(L_ref, L)

% �ڲ�ָ��
% pdist(A)��þ���A�и���������֮����໥���룬squareform()��һά����任Ϊ��Ӧ��M*N����ʽ
distM = squareform(pdist(X_scale)); %����Ⱦ���
DI = dunn_index(K, distM, L)


%% PLOT AREA 
%plotbox_data_prop(X, categories, 'ԭʼ������������');
plot_samples_center(X_scale, U, L, K);       % ������&���ĵ�ռ�ֲ�





