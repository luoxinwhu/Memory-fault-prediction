function DI=dunn_index(clusters_number,distM,L)   
%%%Dunn's index for clustering compactness and separation measurement
% dunns(clusters_number,distM,ind)
% clusters_number = Number of clusters 
% distM = Dissimilarity matrix
%  L   = class labels which are given by "k-mean algorithm" for each data sample points 

i=clusters_number;
denominator=[];

for i2=1:i
    indi=find(L==i2);
    indj=find(L~=i2);
    x=indi;
    y=indj;
    temp=distM(x,y);
    denominator=[denominator;temp(:)];
end

num=min(min(denominator)); 
neg_obs=zeros(size(distM,1),size(distM,2));

for ix=1:i
    indxs=find(L==ix);
    neg_obs(indxs,indxs)=1;
end

dem=neg_obs.*distM;
dem=max(max(dem));

DI=num/dem;
end