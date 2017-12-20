function plot_heatMap_scatter( X, L, K )
%UNTITLED3 �˴���ʾ�йش˺�����ժҪ
%   X ԭʼ���ݼ�

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
ylabel(colorbar, '���1�ĺ������');
title({['���ڸ�˹��Ͼ���ģ�͵�����������ͼ']; ['�����Ŀ��', num2str(K), '  ��ͼ�о�Ϊ��һ��ֵ��']});
grid on;

end

