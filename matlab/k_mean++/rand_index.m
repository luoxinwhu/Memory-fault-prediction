function [AR,RI,MI,HI]=rand_index(L1,L2)
%RANDINDEX - calculates Rand Indices to compare two partitions
% 1.    ARI = RANDINDEX(L1,L2)                   where L1,L2 are vectors listing the class membership, returns the "Hubert & Arabie" adjusted Rand index
% 2.    [AR,RI,MI,HI] = RANDINDEX(L1,L2)     returns the adjusted Rand index¡¢the unadjusted Rand index¡¢Mirkin's index ¡¢Hubert's index.
%
% See L. Hubert and P. Arabie (1985) "Comparing Partitions" Journal of Classification 2:193-218


if nargin < 2 | min(size(L1)) > 1 | min(size(L2)) > 1
   error('RandIndex: Requires two vector arguments')
   return ;
end

C=Contingency(L1,L2);       %form contingency matrix

n=sum(sum(C));
nis=sum(sum(C,2).^2);		%sum of squares of sums of rows
njs=sum(sum(C,1).^2);		%sum of squares of sums of columns

t1=nchoosek(n,2);              %total number of pairs of entities
t2=sum(sum(C.^2));           %sum over rows & columnns of nij^2
t3=.5*(nis+njs);

%Expected index (for adjustment)
nc=(n*(n^2+1)-(n+1)*nis-(n+1)*njs+2*(nis*njs)/n)/(2*(n-1));

A= t1+t2-t3;    %no. agreements
D= -t2+t3;		%no. disagreements

if t1==nc
   AR=0;                          %avoid division by zero; if k=1, define Rand = 0
else
   AR=(A-nc)/(t1-nc);		%adjusted Rand - Hubert & Arabie 1985
end

RI=A/t1;              %Rand 1971		%Probability of agreement
MI=D/t1;             %Mirkin 1970	  %p(disagreement)
HI=(A-D)/t1;        %Hubert 1977	%p(agree)-p(disagree)

function Cont=Contingency(Mem1,Mem2)
    if nargin < 2 | min(size(Mem1)) > 1 | min(size(Mem2)) > 1
       error('Contingency: Requires two vector arguments')
       return;
    end
    Cont=zeros(max(Mem1),max(Mem2));
    disp(size(Mem1, 1));
    for i = 1: size(Mem1,1)
       Cont(Mem1(i),Mem2(i))=Cont(Mem1(i),Mem2(i))+1;
    end
end


end

