function [L,U] = kmeans(X,K)
%KMEANS Cluster multivariate data using the k-means++ algorithm.
% X     (matrix, m_properties * n_samples)  ���ݼ���ÿһ����һ��������
% K     (scalar) �ص���Ŀ
% L     (vector, 1*n_samples)�����ÿ�����������������
% U    (matrix, n_samples*K) �����������γɵ����ĵ㣨һ�б�ʾһ�����ĵ㣩

L = [];
L1 = 0;

while length(unique(L)) ~= K    % �Եõ�k������Ϊ����������unique������ȡһ�������еĲ�ͬ��ֵ
    
    % The k-means++ initialization
    
    U = X(:,1+round(rand*(size(X,2)-1)));  % U�����ĵ㼯�ϣ�size(X,2)�����ݼ���X�����ݵ����Ŀ
    L = ones(1,size(X,2));
    for i = 2:K
        D = X-U(:,L); %D�����ݵ������ĵ�֮��ľ��뼯��
        % dot(A,B,r) ��������ĵ�ˣ�����ڵ�rά����ӣ�dot(D,D,1) = D(1,:, 1)*D(1,:,1) +D(2,:,1)*D(2,:,1)+...
        % cumsum�ۼӺ��� cumsum(a,b,c) = [a, a+b, a+b+c];
        D = cumsum(sqrt(dot(D,D,1)));   %��ÿ�����ݵ������ĵ�ľ���һ���ۼ�
        
        if D(end) == 0, U(:,i:K) = X(:,ones(1,K-i+1)); return; end
        U(:,i) = X(:,find(rand < D/D(end),1));  %find�ڶ���������ʾ���ص���������Ŀ
        [~,L] = max(bsxfun(@minus,2*real(U'*X),dot(U,U,1).'));  %��ÿ�����ݵ���з��࣬~��ռλ��
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
