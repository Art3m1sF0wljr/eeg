function [X] = L1minim(b,A,KK,lambda)
% Optimal solution to A*x=b, using regularization,
% minimizing the L_1 norm ||x||_1
% The idea is putting to zero the raws of the matrix A corresponding to low
% elements of the L_2 solution x.
% INPUT
% A,b: matrix A (MxD, with M<D, M number of electrodes, D number of dipole
% sources) and recorded data b=b(t) (matrix MxT, T indicating the samples
% of time)
% p: index of the Lp norm (usually, p=1)
% KK: iterations 
% lambda: regularization parameter (it is summed to AA' to reduce the
% condition number)
% OUTPUT
% X: matrix DxM, such that x=X*b 



warning off
X=[];
denom = (A*A'+lambda*eye(size(A*A'))); 
x1 = A'/denom;
good=1:size(A,2);
N=floor(length(good)/KK);
X=[];
for k = 1:KK % Iterate
    x=x1*b;
    X= x;
    [~,I]=sort(abs(x(good)));
    bad=good(I(1:N));
    good=setdiff(good,bad);

    B=A(:,good);
    lambda=max(eig(B*B')/1000);
    denom = (B*B'+lambda*eye(size(B*B')));
    x1=zeros(size(x1));
    x1(good,:) = B'/denom;
    x1(bad,:)=0;
    %Proximal point algoritm
    [x1,info] = prox_l1(x1,lambda);
    
end


end

