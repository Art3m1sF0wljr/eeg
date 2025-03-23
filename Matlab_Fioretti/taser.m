function [X,activeSources,allOffSources] = taser(record,A,KK,lambda,consecThreshold,overlap)
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
% overlap: numbero of overlapped samples
% OUTPUT
% X: matrix DxM, such that x=X*b
% activeSources: sources that are active after the method TASER
% allOffSources: sources that are active just for the interval of time that
% the user want to verify


warning off
intervalLength = round(consecThreshold);
numIntervals = floor((size(record, 2) - intervalLength) / (intervalLength - overlap)) + 1;
X = zeros(size(A, 2), size(record, 2));  % Inizializza una matrice per i risultati
activeSources = zeros(size(A, 2), 1);
oldActive = [];
allOffSources = [];
Sources_tot = [];

for i = 1:numIntervals
    % Calcola gli indici per l'intervallo corrente
    startIdx = (i - 1) * (intervalLength - overlap) + 1;
    endIdx = startIdx + intervalLength - 1;

    % Estrai l'intervallo corrente dal segnale EEG
    interval = record(:, startIdx:endIdx);

    % Esegui L1minim_T sull'intervallo corrente
    [X_interval, activeSources_interval] = taser_segment(interval, A, KK, lambda,consecThreshold);
    newActive = setdiff(activeSources_interval,oldActive);
    oldActive = activeSources_interval;
    Sources_tot = [Sources_tot activeSources_interval];
    Sources_tot = unique(Sources_tot);
    offSources = setdiff(Sources_tot, activeSources_interval);
    allOffSources =  [allOffSources offSources];
    allOffSources = unique(allOffSources);

    % Aggiorna i risultati globali
    X(:, startIdx:endIdx) = X_interval;
    activeSources = Sources_tot;
    activeSources = unique(activeSources);
end
end

% I dipoli che sono presenti in offSources sono in teoria dipoli
% che si sono spenti dopo circa 100ms (perchè considero s = 0.1).
% Per testare l'attivazione di dipoli per durate minori in tempo
% basta cambiare il valore di s.
