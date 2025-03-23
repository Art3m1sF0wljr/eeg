function [X,activeSource,badSources] = taser_segment(b,A,KK,lambda,consecThreshold)
% Optimal solution to A*x=b, using regularization,
% minimizing the L_1 norm ||x||_1
% The idea is putting to zero the raws of the matrix A corresponding to low
% elements of the L_2 solution x.
% INPUT
% A,b: matrix A (MxD, with M<D, M number of electrodes, D number of dipole
% sources) and recorded data b=b(t) (matrix MxT, T indicating the samples
% of time)
% KK: iterations
% lambda: regularization parameter (it is summed to AA' to reduce the
% condition number)
% consecThreshold: number of consecutive active samples
% OUTPUT
% X: matrix DxM, such that x=X*b
% activeSources: sources that are active after the method TASER
% badSources: sources that are not active after the method TASER


warning off
% Inizializza un vettore per tenere traccia dei contatori di attivazione
activationCounters = zeros(1, size(A, 2));

% Inizializza vettori per tenere traccia dei dipoli attivi consecutivi
activeSource = zeros(1, size(A, 2));
badSources = zeros(1, size(A, 2));

denom = (A*A'+lambda*eye(size(A*A')));
x1 = A'/denom;
good=1:size(A,2);
N=floor(length(good)/KK);
X=[];



for k = 1:KK % IRLS
    X = x1*b;
    [~,I]=sort(abs(mean(X(good,:),2)));
    bad=good(I(1:N)); %Vanno a 0 quelli con energia minore
    good=setdiff(good,bad); %Tolgo gli ultimi con energia minore dopo aver fatto la media lungo le colonne di x e il valore assoluto
    % Ripeto i calcoli
    B=A(:,good);
    lambda=max(eig(B*B'))/1000;
    denom = (B*B'+lambda*eye(size(B*B')));
    x1=zeros(size(x1));
    x1(good,:) = B'/denom;
    x1(bad,:)=0;
    %proximal point algoritm
    [x1,~] = prox_l1(x1,lambda);
end
for i = 1:3:size(A, 2)
    for j = 1:consecThreshold
        if (abs(X(i, j)) > 0 || abs(X(i + 1, j)) > 0 || abs(X(i + 2, j)) > 0)
            activationCounters(round((i + 1) / 3)) = activationCounters(round((i + 1) / 3)) + 1;
        else
            activationCounters(round((i + 1) / 3)) = 0;
        end

        % Controllo sulla finestra temporale
        if activationCounters(round((i + 1) / 3)) >= consecThreshold
            activeSource(round((i + 1) / 3)) = round((i + 1) / 3);
        else
            badSources(round((i + 1) / 3)) = round((i + 1) / 3);
        end
    end
end

activeSource = find(activeSource);
badSources = find(badSources);
end

