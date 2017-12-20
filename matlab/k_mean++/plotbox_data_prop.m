function plotbox_data_prop( data, categories, figure_name)
% 画出测试数据的box可视化界面
figure();
boxplot(data, 'orientation', 'horizontal', 'labels',categories); %设定x绘制在x轴
title(figure_name);
xlabel('属性值', 'FontSize', 12);
grid on;

end

