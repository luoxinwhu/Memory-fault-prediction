function plotbox_data_prop( data, categories, figure_name)
% �����������ݵ�box���ӻ�����
figure();
boxplot(data, 'orientation', 'horizontal', 'labels',categories); %�趨x������x��
title(figure_name);
xlabel('����ֵ', 'FontSize', 12);
grid on;

end

