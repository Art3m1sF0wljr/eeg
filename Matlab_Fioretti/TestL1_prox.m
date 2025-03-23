%Test sul metodo TASER mediante la simulazione dell'attivazione di alcuni
%dipoli nel tempo.



clear all;close all

% Caricamento della libreria contenente il modello della testa. Selezionare
% la prima o la seconda commentando adeguatamente le righe che non sono di
% interesse.

load("HArtMuT_mix_Colin27_small.mat");Source_modelgrid = 0; %Libreiria consigliata, che permette di attuare al meglio la localizzazione delle sorgenti
% load("HArtMuT_NYhead_small.mat");Source_modelgrid = 0; 
% load("HArtMuT_mix_Colin27_sourcemodelgrid.mat");Source_modelgrid = 1; 

if(Source_modelgrid)
    Nsources=3*size(HArtMuT.sourcemodelgrid.leadfield,2);

else
    Nsource_cortex=3*size(HArtMuT.cortexmodel.leadfield,2);
    Nsource_artefacts = 3*size(HArtMuT.artefactmodel.leadfield,2);
    Nsources = Nsource_artefacts + Nsource_cortex;
end




% Creo una superficie che interpoli le posizioni degli elettrodi
S = scatteredInterpolant(HArtMuT.electrodes.chanpos(:,1),HArtMuT.electrodes.chanpos(:,2),HArtMuT.electrodes.chanpos(:,3),'natural');
% Aggiungo punti lungo le direzioni x e y, usando S per valutare il valore
% di z corrispondente sullo scalpo
minX=min(HArtMuT.electrodes.chanpos(:,1));maxX=max(HArtMuT.electrodes.chanpos(:,1));minY=min(HArtMuT.electrodes.chanpos(:,2));maxY=max(HArtMuT.electrodes.chanpos(:,2));
[xx,yy]=meshgrid(linspace(minX,maxX,20),linspace(minY,maxY,20));zz=S(xx,yy);
% controllo quali punti (x,y) sono dentro al convex hull della testa
tess = convhulln(HArtMuT.electrodes.chanpos(:,1:2));in = inhull([xx(:) yy(:)],HArtMuT.electrodes.chanpos(:,1:2),tess);
bad=find(in==0);
zz(bad)=nan;
Nch=length(HArtMuT.electrodes.label);
%Simulo l'utilizzo dei canali presenti nella convenzione standard 10-20.
ch_name = {'Fp1'; 'Fp2'; 'F7'; 'F3'; 'Fz'; 'F4'; 'F8'; 'C3'; 'Cz'; 'C4'; 'T3'; 'T4'; 'T5'; 'T6'; 'P3'; 'Pz'; 'P4'; 'O1'; 'O2'; 'A1'; 'A2'};
k = 1;
for i=1:length(ch_name)
    for j=1:Nch
        if(strcmp(ch_name{i},HArtMuT.electrodes.label{j}))
            good_ch(k)=i;
            good(k)=j;
            position(k,:)=HArtMuT.electrodes.chanpos(j,:);
            k=k+1;
        end
    end
end
clear ch_name;
ch_name=HArtMuT.electrodes.label(good);

%% Castruzione della matrice A

% Simulo l'attivazione di alcune sorgenti
good_dipoles=(1:Nsources/3)';
Nch=size(ch_name,1);
A = zeros(Nch, Nsources);
if(Source_modelgrid)
     n = 1;
    for i=1:Nsource/3
        A(:,n:n+2) = HArtMuT.sourcemodelgrid.leadfield(good,good_dipoles(i),:);
        n = n + 3;
    end
    positions = HArtMuT.sourcemodelgrid.pos;
else
    good_dipoles_cortex=(1:Nsource_cortex/3)';
    good_dipoles_artefact=(1:Nsource_artefacts/3)';
    n = 1;
    for i=1:Nsource_cortex/3
        A(:,n:n+2) = HArtMuT.cortexmodel.leadfield(good,good_dipoles_cortex(i),:);
        n = n + 3;
    end
    for i=1:Nsource_artefacts/3
        A(:,n:n+2) = HArtMuT.artefactmodel.leadfield(good,good_dipoles_artefact(i),:);
        n = n + 3;
    end
    positions = cat(1,HArtMuT.cortexmodel.pos,HArtMuT.artefactmodel.pos);
end

% scelgo un certo numero di sorgenti distribuite in modo random
for j = 1:500
active_sources=5;Xsimulated=[rand(active_sources,1);zeros(Nsources-active_sources,1)];
dipole_label = cat(1,HArtMuT.cortexmodel.labels,HArtMuT.artefactmodel.labels);
AS=randperm(Nsources);Xsimulated=Xsimulated(AS); %Queste prime tre righe permettono di simulare l'attivazione casuale di 5 sorgenti, questo è stato fatto per poter testare TASER facendo 500 prove. Nel caso si vogliono testare solo alcune sorgenti specifiche decommentare le righe successive 


%Simulazione dell'attivazione di 5 dipoli specifici, indicati dal label
%corrispondente.


% for i = 1:size(dipole_label,1)
%     if(strcmp(dipole_label{i},'Left Lateral Occipital Cortex, inferior division'))
%         tmp = Xsimulated(1);
%         Xsimulated(1) = 0;
%         Xsimulated(i)=tmp;
%     elseif(strcmp(dipole_label{i},'Right Lateral Occipital Cortex, inferior division'))
%         tmp = Xsimulated(2);
%         Xsimulated(2) = 0;
%         Xsimulated(i)=tmp;
%     elseif(strcmp(dipole_label{i},'Muscle_Occipitiofrontalis_OccipitalBelly'))
%         tmp = Xsimulated(3);
%         Xsimulated(3) = 0;
%         Xsimulated(i)=tmp;
%     elseif(strcmp(dipole_label{i},'EyeCornea_left_vertical'))
%         tmp = Xsimulated(4);
%         Xsimulated(4) = 0;
%         Xsimulated(i)=tmp;
%     elseif(strcmp(dipole_label{i},'EyeCornea_right_vertical'))
%         tmp = Xsimulated(5);
%         Xsimulated(5) = 0;
%         Xsimulated(i)=tmp;
%     end
% end

iter=100;
lambda=max(eig(A*A'))/1000;


% simulazione di un segnale variabile nel tempo

fs=256;%Campiono a 256Hz
fine = 1;
tt=-fine:1/fs:fine;
ss=.02;
g=diff(exp(-tt.^2/2/ss^2));
len_epoch=length(tt)-1;
nn=3;
vv=[ones(nn,1);zeros(len_epoch-nn,1)];
Xsim=zeros(Nsources,len_epoch);
qq=find(Xsimulated>0);
for i=1:length(qq)
   Xsim(qq(i),:)=Xsimulated(qq(i))*conv(vv(randperm(len_epoch)),g,'same'); 
end
fpass = 70;
s = 0.1; %secondi simulati
b=A*Xsim;
b = resample(b,100,fs,Dimension = 2); %Sottocampionamento a 100Hz
fs = 100;
b = lowpass(b,fpass,fs);
tt=0:1/fs:fine-1/fs;

% Simulazione di un "blink" nell'EEG
tempo_blink = fine*rand(1,fine); % Tempo in cui si verifica il blink 
durata_blink = 0.1; % Durata del "blink"
ampiezza_blink = 1e-4; % Ampiezza dell'artefatto del "blink"
indice_blink = round(tempo_blink * fs);
durata_campioni_blink = round(durata_blink * fs);
for i = 1:fine
    b([1,2,7],(indice_blink(:,i) + round(durata_campioni_blink/4))) = ampiezza_blink;
    b([1,2,7],(indice_blink(:,i) + round(durata_campioni_blink/2))) = -ampiezza_blink;
end

% Simulazione del ritmo alpha sovrapposto al segnale di base negli
% elettrodi occipitali e parietali

tempo_alpha = round(fine/2)*rand(1,round(fine/2)); %Tempo in cui si verifica il ritmo alpha
frequenza_alpha = 10; % Frequenza tipica del ritmo alpha (può variare tra 8-12 Hz)
ampiezza_alpha = 35e-6; % Ampiezza del ritmo alpha
durata_alpha = s;
indice_alpha = round(tempo_alpha*fs);
durata_campioni_alpha = round(durata_alpha * fs);
ritmo_alpha = ampiezza_alpha * sin(2 * pi * frequenza_alpha * tt);

for i = 1:round(fine/2)
    
    % Combinazione del ritmo alpha con il segnale di base
    b(15:19,indice_alpha(:,i):indice_alpha(:,i) + durata_campioni_alpha-1) = b(15:19,(indice_alpha(:,i):indice_alpha(:,i) + durata_campioni_alpha-1)) + ritmo_alpha((indice_alpha(:,i):indice_alpha(:,i) + durata_campioni_alpha-1));

end

% Simulazione del ritmo beta sovrapposto al segnale di base
tempo_beta = round(fine)*rand(1,round(fine)); %Tempo in cui si verifica il ritmo alpha
frequenza_beta = 20; % Frequenza tipica del ritmo beta (può variare tra 13-30 Hz)
ampiezza_beta = 5e-6; % Ampiezza del ritmo beta circa un microVolt
durata_beta = 2*s;
indice_beta = round(tempo_beta*fs);
durata_campioni_beta = round(durata_beta * fs);
ritmo_beta = ampiezza_beta * sin(2 * pi * frequenza_beta * tt);

for i = 1:round(fine)
    
    % Combinazione del ritmo beta con il segnale di base
    b([1:10 18:19],indice_beta(:,i):indice_beta(:,i) + durata_campioni_beta-1) = b([1:10 18:19],(indice_beta(:,i):indice_beta(:,i) + durata_campioni_beta-1)) + ritmo_beta((indice_beta(:,i):indice_beta(:,i) + durata_campioni_beta-1));

end


% mostra_segnali(b/6/std(b(:)),'k','',fs);grid on;xlabel('Tempo (s)');xlim([0 fine]);
% set(gca,'YTick',1:Nch);set(gca,'YTickLabel',ch_name);title('Simulated signal')


%% Minimizzazione su tutto il segnale

% figure;
F = scatteredInterpolant(HArtMuT.electrodes.chanpos(good,1),HArtMuT.electrodes.chanpos(good,2),HArtMuT.electrodes.chanpos(good,3),mean(abs(b(:,1:round(fs*s))),2),'natural');
% surf(xx,yy,zz,F(xx,yy,zz),'EdgeColor','none');alpha(.6);xlabel('x');ylabel('y');zlabel('z');axis equal;hold on;
overlap = 1;
consecThreshold = fs*s;
% campMax = size(b,2);
campMax = 100;
[X1,activeSources,allOffSources] = taser(b(:,1:campMax),A,iter,lambda,consecThreshold,overlap);
   
%Decommentare le righe sottostanti se si vogliono vedere i plot delle
%sorgenti

% plot3(positions(activeSources,1),positions(activeSources,2),positions(activeSources,3),'ro','MarkerSize',3,'LineWidth',3);hold on;
% plot3(positions(allOffSources,1),positions(allOffSources,2),positions(allOffSources,3),'ko','MarkerSize',3,'LineWidth',3);hold on;
% plot3(positions(round(qq/3)+1,1),positions(round(qq/3)+1,2),positions(round(qq/3)+1,3),'bo','MarkerSize',4,'LineWidth',4)


figure;
b1 = A*X1;
mostra_segnali(b/6/std(b(:)),'k','',fs);mostra_segnali(b1/6/std(b1(:)),'r','',fs);grid on;xlabel('Tempo (s)');xlim([0 fine]);
set(gca,'YTick',1:Nch);set(gca,'YTickLabel',ch_name);title('Simulated signal')
legend('EEG simulated signal', 'EEG recovered signal')


%% Test


X_mne = MNE(A,b);
est = A*X_mne; %segnale stimato MNE
est2 = A*X1;
MSE_MNE(j)=(std(est-b)/std(b))^2;
disp(['MSE_MNE = ' num2str(MSE_MNE*100) '%'])
MSE_TASER(j)=(std(est2-b(:,1:campMax))/std(b(:,1:campMax)))^2;
disp(['MSE_TASER = ' num2str(MSE_TASER*100) '%'])
numero_sorg(j) = size(allOffSources,2); %Vettore contenente il numero di sorgenti che possono essere considerate "buone" ad ogni prova.

index = find(Xsimulated);
disp(dipole_label(round(index/3)+1))
end
