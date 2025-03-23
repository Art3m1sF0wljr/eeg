close all;clear all
% carica la libreria
load('DipoleField');
% Carica l'EEG in formato edf
[fname,fdir]=uigetfile('*.edf','select the file');
[hdr, recrd] = edfread([fdir fname]);
ch_name=hdr.label;
% Cerca la posizione dei canali EEG in base al loro nome riportato nel file
% edf
k=1;
for i=1:length(ch_name)% Loop over each element in ch_name
    for j=1:97 % Loop from 1 to 97 (assuming 97 electrodes)
        if(strcmp(ch_name{i},elec.label{j}))  % Check if the channel name matches the electrode label
            good_ch(k)=i; % Store the index of the matching channel
			good(k)=j; % Store the index of the matching electrode
            position(k,:)=elec.elecpos(j,:);% Store the electrode's position
            k=k+1;% Increment index for next match
        end
    end
end
clear ch_name;ch_name=elec.label(good);
record=recrd(good_ch,:);
Nch=size(record,1);
fs=hdr.samples(10);N=fs;

% Visualizzazione leadfield di un dipolo (semplice applicazione per
% valutare se un dipolo con una certa posizione fornisce un potenziale
% superficiale coerente con le aspettative)  
figure
% Creo una superficie che interpoli le posizioni degli elettrodi
S = scatteredInterpolant(elec.elecpos(:,1),elec.elecpos(:,2),elec.elecpos(:,3),'natural');
% Aggiungo punti lungo le direzioni x e y, usando S per valutare il valore
% di z corrispondente sullo scalpo
% Define the bounds of the scalp (electrode positions)
minX=min(elec.elecpos(:,1));
maxX=max(elec.elecpos(:,1));
minY=min(elec.elecpos(:,2));
maxY=max(elec.elecpos(:,2));
% Create a 2D grid for interpolation
[xx,yy]=meshgrid(linspace(minX,maxX,20),linspace(minY,maxY,20));
% Compute interpolated Z-values to map the scalp surface
zz=S(xx,yy);
% controllo quali punti (x,y) sono dentro al convex hull della testa
tess = convhulln(elec.elecpos(:,1:2));
in = inhull([xx(:) yy(:)],elec.elecpos(:,1:2),tess);
% Remove points outside the convex hull
bad=find(in==0);
zz(bad)=nan;

% Creo una funzione che interpola il valore della variabile di interesse
% nella posizione degli elettrodi
Num_dipole=50;% Selected dipole index
str='>^x';% Marker symbols for different orientations
for orientation=1:3
    h(orientation)=subplot(1,3,orientation);
    plot3(DipoleField.pos(:,1),DipoleField.pos(:,2),DipoleField.pos(:,3),'k.');hold on
    plot3(DipoleField.pos(Num_dipole,1),DipoleField.pos(Num_dipole,2),DipoleField.pos(Num_dipole,3),['r' str(orientation)],'MarkerSize',10,'LineWidth',3)    % Mark selected dipole in red
        % Normalize the lead field
	Field=DipoleField.leadfield{Num_dipole}(:,orientation);
	Field=(Field-min(Field(:)))/range(Field(:));
    % Interpolate the field at the electrode positions
	F = scatteredInterpolant(elec.elecpos(:,1),elec.elecpos(:,2),elec.elecpos(:,3),Field,'natural');
    % Mostro la variabile di interesse interpolata sullo scalpo
    surf(xx,yy,zz,F(xx,yy,zz),'EdgeColor','none');alpha(.6);xlabel('x');ylabel('y');zlabel('z');axis equal
end
% Link all subplots to have the same camera view
hlink = linkprop(h,{'CameraPosition','CameraUpVector'}); view(2)

% Associo ciascun dipolo all'elettrodo più vicino
pos_good_dipoles=DipoleField.pos(DipoleField.inside,:);
DistMat=pdist2(pos_good_dipoles,position);
[m,I]=min(DistMat');


figure;
surf(xx,yy,zz,ones(size(zz)),'EdgeColor','none');alpha(.5);hold on
str1='o<>sx+';str2='krgbc';
for i=1:length(position)
    plot3(position(i,1),position(i,2),position(i,3),[str2(mod(i,5)+1) str1(mod(i,6)+1)],'MarkerSize',10);
    vv=pos_good_dipoles(find(I==i),:);
    plot3(vv(:,1),vv(:,2),vv(:,3),[str2(mod(i,5)+1) str1(mod(i,6)+1)],'MarkerSize',2);
end

% Stima delle attività dei dipoli che hanno prodotto il segnale EEG registrato
% Construct Leadfield Matrix A
%The Leadfield Matrix A is a mathematical representation of how the activity of brain dipoles is projected onto the EEG sensors
%(electrodes). It tells us how each dipole's activity (in the brain) contributes to the recorded EEG signal at each electrode.
% The elements of this matrix represent the contribution of each dipole component (x, y, and z) to each electrode measurement.
Nsource=3*sum(DipoleField.inside); % Total number of dipole components (3 per dipole)
%The total number of dipole components, calculated by multiplying the number of active dipoles by 3 (each dipole has 3 components: x, y, z).
good_dipoles=find(DipoleField.inside==1); % Indices of active dipoles
%Finds the indices of the active dipoles. These are the dipoles that contribute to the EEG signals.
A = zeros(Nch, Nsource);
%Nch: The number of electrodes (channels) in the EEG setup.
%Each column in A represents the contribution of a single dipole component (x, y, or z) to all electrodes.
n = 1;
for i=1:Nsource/3%Loops through each dipole
    A(:,n:n+2) = DipoleField.leadfield{good_dipoles(i)}(good,:);% Store leadfield for each dipole
    %Contains the leadfield matrix for dipole i (how dipole activity maps to EEG sensors
    n = n + 3; % Move to the next 3 columns (x, y, z components)
end
%A=21x24570 so it's highly non square
% stima delle attività dei dipoli, minimizzando lo scarto quadratico con un
% termine di regolarizzazione che penalizza l'energia 
L=eig(A*A');%gives an idea of signal power
%(which is essentially the covariance of the EEG data). The eigenvalues give an idea of the signal strength or variance in the leadfield matrix.
%Large eigenvalues indicate dominant signal directions (i.e., major source contributions).
%Small eigenvalues correspond to noise or weak sources
Lambda=max(L)/1000;%Regularization parameter (prevents overfitting by controlling noise
denom = (A*A'+Lambda*eye(size(A*A')));%Adds a regularization term to stabilize inversion
%This adds regularization to the matrix A * A'. The term Lambda * eye(size(A * A')) is a diagonal matrix that adds a small value to the diagonal elements, preventing instability in the matrix inversion process.
w = A' / denom;% Compute the inverse solution (weights)
%These weights will be used to map EEG signals back to dipole activity. It can be thought of as a reverse projection from EEG measurements to brain sources (dipoles).

% stima del segnale
est=(A*w)*record;%estimated EEG signal
% gives a matrix that represents the mapping from dipole activity to EEG measurements.
% Errore nella ricostruzione del segnale come somma delle attività dei
% dipoli stimati
MSE=(std(record-est)/std(record))^2;
disp(['MSE = ' num2str(MSE*100) '%'])
figure;
hl(1)=subplot(211);
mostra_segnali(record/6/std(record(:)),'k','',fs);
mostra_segnali(est/6/std(record(:)),'r','',fs);
title('EEG signal')
legend('raw EEG','estimated EEG')
set(gca,'YTick',[1:Nch]);set(gca,'YTickLabel',ch_name)
for ch=1:Nch
    select_dipoles=find(I==ch);
    select_col=sort([3*(select_dipoles-1)+1 3*(select_dipoles-1)+2 3*(select_dipoles-1)+3]);
    vv=(A(:,select_col)*w(select_col,:))*record;
    sig(ch,:)=vv(ch,:);
end

hl(2)=subplot(212);
mostra_segnali(sig/6/std(sig(:)),'k','',fs);
set(gca,'YTick',[1:Nch]);set(gca,'YTickLabel',ch_name);title('Selective channels, recording signals coming from closest dipoles, eliminating crosstalk')
linkaxes(hl,'xy')