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
minX=min(elec.elecpos(:,1));maxX=max(elec.elecpos(:,1));
minY=min(elec.elecpos(:,2));maxY=max(elec.elecpos(:,2));
% Create a 2D grid for interpolation
[xx,yy]=meshgrid(linspace(minX,maxX,20),linspace(minY,maxY,20));
% Compute interpolated Z-values to map the scalp surface
zz=S(xx,yy);
% controllo quali punti (x,y) sono dentro al convex hull della testa
tess = convhulln(elec.elecpos(:,1:2));in = inhull([xx(:) yy(:)],elec.elecpos(:,1:2),tess);
bad=find(in==0);
zz(bad)=nan;

% Creo una funzione che interpola il valore della variabile di interesse
% nella posizione degli elettrodi
Num_dipole=50;
str='>^x';
for orientation=1:1
    h(orientation)=subplot(1,3,orientation);
    plot3(DipoleField.pos(:,1),DipoleField.pos(:,2),DipoleField.pos(:,3),'k.');hold on
    plot3(DipoleField.pos(Num_dipole,1),DipoleField.pos(Num_dipole,2),DipoleField.pos(Num_dipole,3),['r' str(orientation)],'MarkerSize',10,'LineWidth',3)
    Field=DipoleField.leadfield{Num_dipole}(:,orientation);Field=(Field-min(Field(:)))/range(Field(:));
    F = scatteredInterpolant(elec.elecpos(:,1),elec.elecpos(:,2),elec.elecpos(:,3),Field,'natural');
    % Mostro la variabile di interesse interpolata sullo scalpo
    surf(xx,yy,zz,F(xx,yy,zz),'EdgeColor','none');alpha(.6);xlabel('x');ylabel('y');zlabel('z');axis equal
end
hlink = linkprop(h,{'CameraPosition','CameraUpVector'}); view(2)

% Associo ciascun dipolo all'elettrodo più vicino
pos_good_dipoles=DipoleField.pos(DipoleField.inside,:);
DistMat=pdist2(pos_good_dipoles,position);[m,I]=min(DistMat');