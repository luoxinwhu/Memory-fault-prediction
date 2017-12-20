function save_xls_to_mat( excel_name, X_range, categories_range, mat_name )
%SAVE_MAT 读取Excel文件内容，并保存成mat文件
%   excel_name              Excel文件名
%   X_range                  数据单元格的范围
%   categories_range    种类名单元格范围
%   mat_name                mat文件名

X = xlsread(excel_name,X_range);
[~, categories] = xlsread(excel_name, categories_range);
save(mat_name , 'X', 'categories');
end

