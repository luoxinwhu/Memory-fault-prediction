function plot_samples_center( X, U, L, K)
% plot_samples ���ƾ������������������ĵ�ֲ����
% X ԭʼ������ ��n_samples * n_properties��
% U ͨ��k-means++ �㷨�õ������ĵ㼯 ��K * n_properties��
% L ͨ��k-means++ �㷨�õ�������ǩ�� ��n_samples * 1��
% K �������Ŀ


        
P1 = figure;
symb = {'b*', 'g*', 'y*', 'm*', 'c*', 'k*'};
MarkFace = {[1,0,0], [0.8, 0, 1], [1, 0.5, 0], [0.8, 0, 0], [0.4, 1, 0], [0, 0.9, 0.5]};
for k=1:K
    clust = find(L==k);
    plot(X(clust, 3), X(clust, 5), symb{k});    % ѡȡ��3��5�����Ի�������ɢ��ͼ
    legend_str{k}=['���',num2str(k)];   %����ÿһ��legend�ı�
    hold on;
end
plot(U(:, 3), U(:, 5), 'sr', 'MarkerSize', 10, 'MarkerFace', MarkFace{k});   %�������ĵ�

% �趨ͼ����ʽ
title('����������ĵ�ɢ��ͼ');
xlabel('����3 Excess kurtosis of the integrated profile');
ylabel('����5 Mean of the DM-SNR curve');
xlim([-0.2, 1.2]);
ylim([-0.2, 1.2]);
legend(legend_str ,'���ĵ�'); %ͳһ���ͼ��
end

