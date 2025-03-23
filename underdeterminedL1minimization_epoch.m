function [X,L2norm,L1norm] = underdeterminedL1minimization_epoch(b,A,KK,lambda)
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
% % EXAMPLE 1
% S=2;M=2;N=5;A=randn(M,N);xx=[1+rand(S,1);zeros(N-S,1)];b=A*xx;iter=5;
% [X,L2norm,L1norm] = underdeterminedL1minimization(b,A,iter,max(eig(A*A'))/1000);
% figure;subplot(2,1,1);plot(xx,'k');hold on;plot(X(:,round(end/5)),'r');plot(X(:,round(end/2)),'m');plot(X(:,end),'g');legend('simulated',[num2str(iter/5) ' iter'],[num2str(iter/2) ' iter'],[num2str(iter) ' iter'])
% title('Imposing some values to be zero')
% subplot(2,1,2);plot(L2norm);hold on;plot(L1norm);xlabel('iterations');legend('L_2 norm','L_1 norm')
% [X,L2norm,L1norm] = IRLS0(b,A,1,iter,max(eig(A*A'))/1000);
% figure;subplot(2,1,1);plot(xx,'k');hold on;plot(X(:,round(end/5)),'r');plot(X(:,round(end/2)),'m');plot(X(:,end),'g');legend('simulated',[num2str(iter/5) ' iter'],[num2str(iter/2) ' iter'],[num2str(iter) ' iter'])
% title('IRLS')
% subplot(2,1,2);plot(L2norm);hold on;plot(L1norm);xlabel('iterations');legend('L_2 norm','L_1 norm')
% [X,L2norm,L1norm]  = IRLS2original(A,b,1,.8,iter);
% figure;subplot(2,1,1);plot(xx,'k');hold on;plot(X(:,round(end/5)),'r');plot(X(:,round(end/2)),'m');plot(X(:,end),'g');legend('simulated',[num2str(iter/5) ' iter'],[num2str(iter/2) ' iter'],[num2str(iter) ' iter'])
% title('IRLS original')
% subplot(2,1,2);plot(L2norm);hold on;plot(L1norm);xlabel('iterations');legend('L_2 norm','L_1 norm')
% 
% % EXAMPLE 2


warning off
X=[];
denom = (A*A'+lambda*eye(size(A*A')));x1 = A'/denom;
%x1=pinv(A);
good=1:size(A,2);
N=floor(length(good)/KK);
X=[];
for k = 1:KK % Iterate
    x=x1*b;
    L2norm(k)=sqrt(mean(x(:).^2));
    L1norm(k)=mean(abs(x(:)));
    [~,I]=sort(abs(mean(x(good,:),2)));bad=good(I(1:N));
    %bad=good(find(abs(x(good))<.2*max(abs(x))));
    good=setdiff(good,bad);
    B=A(:,good);lambda=max(eig(B*B'))/1000;
    denom = (B*B'+lambda*eye(size(B*B')));
    x1=zeros(size(x1));
    x1(good,:) = B'/denom;
    x1(bad,:)=0;
    if min(L1norm)==L1norm(k)
       x1best=x1; 
    end
end
X=x1best*b;
end

