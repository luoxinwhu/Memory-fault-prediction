function plot_scatter( X, L, K )
% PLOT_SCATTER ���ƶ�ά����άɢ��ͼ
%  X    ԭʼ������ ��matrix�� n_samples * n_properties��
%  L    ͨ����˹��Ͼ���õ�����ȱ�ǩ����vector, n_samples*1)
%  K     ��˹���ģ���еĴ�����Ŀ

[n_samples, n_properties] = size(X);

% ��þ��������Ŀ
if isscalar(K)
    K_number = K;
else
    K_number = size(K, 1);
end

pr_color = zeros(n_samples, 3);
for k = 1: K_number
    color = [rand, rand, rand];            %������ɫ���������������ɫ
    csize=size(pr_color(L==k,:),1);        % ����ĳһ���������Ŀ
    pr_color(L==k,:)=repmat(color,csize,1);    %������ĳһ���µĵ�Ⱦɫ
end

figure(1);
scatter3(X(:, 1), X(:, 2), X(:, 3), 10, '*');
title('ԭʼ����ɢ��ͼ��ͼ�о�Ϊ��һ��ֵ��');
xlabel('����1');
ylabel('����2');
zlabel('����3');


figure(2);
if n_properties==2
    scatter(X(:, 1), X(:, 2), 10, pr_color);
elseif n_properties>=3
    scatter3(X(:, 1), X(:, 2), X(:, 3), 10, pr_color, 'filled');
end
title({['���ڸ�˹��Ͼ���ģ�͵�����ɢ��ͼ']; ['�����Ŀ��', num2str(K), '  ��ͼ�о�Ϊ��һ��ֵ��']});
xlabel('����1');
ylabel('����2');
zlabel('����3');

end

