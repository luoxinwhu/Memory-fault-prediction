function save_xls_to_mat( excel_name, X_range, categories_range, mat_name )
%SAVE_MAT ��ȡExcel�ļ����ݣ��������mat�ļ�
%   excel_name              Excel�ļ���
%   X_range                  ���ݵ�Ԫ��ķ�Χ
%   categories_range    ��������Ԫ��Χ
%   mat_name                mat�ļ���

X = xlsread(excel_name,X_range);
[~, categories] = xlsread(excel_name, categories_range);
save(mat_name , 'X', 'categories');
end

