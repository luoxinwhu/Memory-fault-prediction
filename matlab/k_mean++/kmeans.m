function [L,U] = kmeans(X,K)
%KMEANS Cluster multivariate data using the k-means++ algorithm.
% X     (matrix, m_properties * n_samples)  数据集，每一列是一个样本点
% K     (scalar) 簇的数目
% L     (vector, 1*n_samples)标记了每个样本点的所属分类
% U    (matrix, n_samples*K) 保存了最终形成的中心点（一列表示一个中心点）

L = [];
L1 = 0;

while length(unique(L)) ~= K    % 以得到k个聚类为结束条件，unique函数获取一个矩阵中的不同的值
    
    % The k-means++ initialization
    
    U = X(:,1+round(rand*(size(X,2)-1)));  % U是中心点集合，size(X,2)是数据集合X的数据点的数目
    L = ones(1,size(X,2));
    for i = 2:K
        D = X-U(:,L); %D是数据点与中心点之间的距离集合
        % dot(A,B,r) 两个矩阵的点乘，结果在第r维度相加，dot(D,D,1) = D(1,:, 1)*D(1,:,1) +D(2,:,1)*D(2,:,1)+...
        % cumsum累加函数 cumsum(a,b,c) = [a, a+b, a+b+c];
        D = cumsum(sqrt(dot(D,D,1)));   %将每个数据点与中心点的距离一次累加
        
        if D(end) == 0, U(:,i:K) = X(:,ones(1,K-i+1)); return; end
        U(:,i) = X(:,find(rand < D/D(end),1));  %find第二个参数表示返回的索引的数目
        [~,L] = max(bsxfun(@minus,2*real(U'*X),dot(U,U,1).'));  %将每个数据点进行分类，~是占位符
    end
    
    % The k-means algorithm.
    while any(L ~= L1) 
        L1 = L;
        for i = 1:K
            l = (L==i); 
            U(:,i) = sum(X(:,l),2)/sum(l); 
        end
        [~,L] = max(bsxfun(@minus,2*real(U'*X),dot(U,U,1).'),[],1);
    end
    
    L = L' ;
    U = U' ;
    
end
