% EstimateSourcces_HArtMuT_L1prox è un codice fatto per testare il metodo
% TASER su un segnale reale. Per prima cosa, una volta lanciato il codice
% verrà chiesto di selezionare il file .edf che contiene il segnale EEG.
clear all
close all
clc

% Caricamento della libreria contenente il modello della testa. Selezionare
% la prima o la seconda commentando adeguatamente le righe che non sono di
% interesse.

load("HArtMuT_mix_Colin27_small.mat");Source_modelgrid = 0; %Libreria consigliata, che permette di attuare al meglio la localizzazione delle sorgenti
% load("HArtMuT_NYhead_small.mat");Source_modelgrid = 0; 
% load("HArtMuT_mix_Colin27_sourcemodelgrid.mat");Source_modelgrid = 1; 


%Selezione del file .edf
[fname,fdir]=uigetfile('*.edf','select the file');
[hdr, recrd] = edfread([fdir fname]);
ch_name=hdr.label;
N_elec = size(HArtMuT.electrodes.chanpos,1);

%Selezione dei soli canali "buoni"
k=1;
for i=1:length(ch_name)
    for j=1:N_elec
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
record=recrd(good_ch,:);
Nch=size(record,1);
fs=hdr.samples(10);
record = resample(record,100,fs,Dimension = 2); %Sottocampionamento a 100Hz come frequenza di campionamento
fs = 100;
N=fs;


S = scatteredInterpolant(HArtMuT.electrodes.chanpos(:,1),HArtMuT.electrodes.chanpos(:,2),HArtMuT.electrodes.chanpos(:,3),'natural');
minX=min(HArtMuT.electrodes.chanpos(:,1));maxX=max(HArtMuT.electrodes.chanpos(:,1));minY=min(HArtMuT.electrodes.chanpos(:,2));maxY=max(HArtMuT.electrodes.chanpos(:,2));
[xx,yy]=meshgrid(linspace(minX,maxX,20),linspace(minY,maxY,20));zz=S(xx,yy);
tess = convhulln(HArtMuT.electrodes.chanpos(:,1:2));
in = inhull([xx(:) yy(:)],HArtMuT.electrodes.chanpos(:,1:2),tess);
bad=find(in==0);
zz(bad)=nan;

%HArtMuT.leadfield è fatto in maniera diversa, cioè non è un array di celle
%ma è un array multidimensionale
 Num_dipole=50;
    str='>^x';

%% Costruzione della matrice A

if(Source_modelgrid) %Questo if permette di usare la libreria contenente il modello della testa in maniera corretta.

    % Stima delle attività dei dipoli che hanno prodotto il segnale EEG
    % registrato
    Nsource=3*size(HArtMuT.sourcemodelgrid.leadfield,2);
    good_dipoles=(1:Nsource/3)';
    A = zeros(Nch, Nsource);
    n = 1;
    for i=1:Nsource/3
        A(:,n:n+2) = HArtMuT.sourcemodelgrid.leadfield(good,good_dipoles(i),:);
        n = n + 3;
    end
    positions = HArtMuT.sourcemodelgrid.pos;
    iter = 100;
    lambda=max(eig(A*A'))/1000;
    s = 0.1; %Secondi simulati
    fpass = 70; %Hz freq banda
    record = lowpass(record,fpass,fs);

    [X] = taser(record(:,1:round(fs*s)),A,iter,lambda,fs*s); % Metodo TASER

     % Decommentare questa parte se si desidera avere i vari plot.

%     figure;
%     
%     for i =1:round(fs*s)
%         plot(1:Nsource/3,A(:,1:3:end)\record(:,i),'k',1:Nsource/3,A(:,2:3:end)\record(:,i),'g',1:Nsource/3,A(:,3:3:end)\record(:,i),'r')
%         grid on;
%         axis([0 size(X,1)/3 -1e7 1e7])
%         pause(1/round(fs*s))
%     end
%     legend('record orientazione x dipolo','record orientazione y dipolo','record orientazione z dipolo')
%    
%     figure;
%
%     for i =1:round(fs*s)
%         plot(1:Nsource/3,X(1:3:end,end-round(fs*s)+i),'k',1:Nsource/3,X(2:3:end,end-round(fs*s)+i),'g',1:Nsource/3,X(3:3:end,end-round(fs*s)+i),'r') %Orient x,y,z
%         grid on;
%         axis([0 size(X,1)/3 -2e5 2e5])
%         pause(1/round(fs*s))
%     end
%     
%     legend([num2str(iter) ' iter x'],[num2str(iter) ' iter y'],[num2str(iter) ' iter z'])

else %Seleziona la maniera corretta di utilizzare la libreria HArtMuT
   

    

    % Stima delle attività dei dipoli che hanno prodotto il segnale EEG
    % registrato

    Nsource=3*size(HArtMuT.cortexmodel.leadfield,2) + 3*size(HArtMuT.artefactmodel.leadfield,2);
    Nsource_cortex=3*size(HArtMuT.cortexmodel.leadfield,2);
    Nsource_artefacts = 3*size(HArtMuT.artefactmodel.leadfield,2);
    good_dipoles = (1:Nsource/3)';
    good_dipoles_cortex=(1:Nsource_cortex/3)';
    good_dipoles_artefact=(1:Nsource_artefacts/3)';
    A = zeros(Nch, Nsource);
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

%% Stima su un'epoca

    iter = 100;
    s = 0.1; %Secondi simulati
    lambda=max(eig(A*A'))/1000; 
    fpass = 40; %Hz freq banda
    record = lowpass(record,fpass,fs);
    
    [X] = taser_segment(record(:,1:round(fs*s)),A,iter,lambda,fs*s); %TASER su una sola epoca
    
% Decommentare questa parte se si desidera avere i vari plot.
    
%     figure;
%     
%     for i =1:round(fs*s)
%         plot(1:Nsource/3,A(:,1:3:end)\record(:,i),'k',1:Nsource/3,A(:,2:3:end)\record(:,i),'g',1:Nsource/3,A(:,3:3:end)\record(:,i),'r')
%        
%         grid on;
%         axis([0 size(X,1)/3 -1e7 1e7])
%         pause(1/round(fs*s))
%     end
%     legend('record orientazione x dipolo','record orientazione y dipolo','record orientazione z dipolo')
%    
% figure;
%     for i =1:round(fs*s)
%         
%         plot(1:Nsource/3,X(1:3:end,end-round(fs*s)+i),'k',1:Nsource/3,X(2:3:end,end-round(fs*s)+i),'g',1:Nsource/3,X(3:3:end,end-round(fs*s)+i),'r') %Orient x,y,z
%         grid on;
%         axis([0 size(X,1)/3 -2e5 2e5])
%         pause(1/round(fs*s))
%     end
%     
%     legend([num2str(iter) ' iter x'],[num2str(iter) ' iter y'],[num2str(iter) ' iter z'])


end


%% Parte comune a tutti i modelli



figure
str = '>^x';
F = scatteredInterpolant(HArtMuT.electrodes.chanpos(good_ch,1),HArtMuT.electrodes.chanpos(good_ch,2),HArtMuT.electrodes.chanpos(good_ch,3),mean(abs(record(:,1:round(fs*s))),2),'natural');
surf(xx,yy,zz,F(xx,yy,zz),'EdgeColor','none');alpha(.6);xlabel('x');ylabel('y');zlabel('z');axis equal;hold on;
for i = 1:round(fs*s)
    for orient = 1:3
        qq=find(X(orient:3:end,end-round(fs*s)+i)>0);
        for Num=1:length(qq)
            meanA=X(qq(Num)*3-(3-orient),end-round(fs*s)+i);
            Num_dipole=round(qq(Num));
            color_level=round(100*(.9-.9*meanA/max(X(:,end-round(fs*s)+i))))/100;
            plot3(positions(good_dipoles(Num_dipole),1),positions(good_dipoles(Num_dipole),2),positions(good_dipoles(Num_dipole),3),str(orient),'color',color_level*[1 1 1],'MarkerSize',1+5*meanA/max(X(:,end-round(fs*s)+i)),'LineWidth',1+2*meanA/max(X(:,end-round(fs*s)+i)))
        end
    end
    pause(1/round(fs*s))
end
% hold on
%  for i = 1:round(fs*s)
%     for orient = 1:3
%         qq=find(A(:,orient:3:end)\record(:,i)>0);
%         for Num=1:length(qq)
%             meanA=abs(A(:,qq(Num)*3-(3-orient))\record(:,i));Num_dipole=round(qq(Num));
%             plot3(positions(good_dipoles(Num_dipole),1),positions(good_dipoles(Num_dipole),2),positions(good_dipoles(Num_dipole),3),'ro','MarkerSize',1+5*meanA/max(A\record(:,i)),'LineWidth',1+2*meanA/max(A\record(:,i)))
%         end
%     end
%     pause(1/round(fs*s))
% end
title('prox_{L1} minimization')

%% Minimizzazione su tutto il segnale
figure;
F = scatteredInterpolant(HArtMuT.electrodes.chanpos(good_ch,1),HArtMuT.electrodes.chanpos(good_ch,2),HArtMuT.electrodes.chanpos(good_ch,3),mean(abs(record(:,1:round(fs*s))),2),'natural');
surf(xx,yy,zz,F(xx,yy,zz),'EdgeColor','none');alpha(.6);xlabel('x');ylabel('y');zlabel('z');axis equal;hold on;
overlap = 0;
consecThreshold = fs*s;
% campMax = size(record,2); %Campione massimo
campMax = 40;
[X1,activeSources,allOffSources] = taser(record(:,1:campMax),A,iter,lambda,consecThreshold,overlap);

plot3(positions(activeSources,1),positions(activeSources,2),positions(activeSources,3),'ro','MarkerSize',3,'LineWidth',3);hold on;
plot3(positions(allOffSources,1),positions(allOffSources,2),positions(allOffSources,3),'ko','MarkerSize',3,'LineWidth',3);


                            