function X = MNE(A,record)
L=eig(A*A');Lambda=max(L)/1000;
denom = (A*A'+Lambda*eye(size(A*A')));
w = A' / denom;

% stima del segnale
X=w*record;
