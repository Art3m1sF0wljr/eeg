function [A] = EstimateSources()
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
for i=1:length(ch_name)
    for j=1:97
        if(strcmp(ch_name{i},elec.label{j}))
            good_ch(k)=i;good(k)=j;
            position(k,:)=elec.elecpos(j,:);
            k=k+1;
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
minX=min(elec.elecpos(:,1));maxX=max(elec.elecpos(:,1));minY=min(elec.elecpos(:,2));maxY=max(elec.elecpos(:,2));
[xx,yy]=meshgrid(linspace(minX,maxX,20),linspace(minY,maxY,20));zz=S(xx,yy);
% controllo quali punti (x,y) sono dentro al convex hull della testa
tess = convhulln(elec.elecpos(:,1:2));in = inhull([xx(:) yy(:)],elec.elecpos(:,1:2),tess);
bad=find(in==0);
zz(bad)=nan;

% Creo una funzione che interpola il valore della variabile di interesse
% nella posizione degli elettrodi
Num_dipole=50;
str='>^x';
for orientation=1:3
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
figure;
surf(xx,yy,zz,ones(size(zz)),'EdgeColor','none');alpha(.5);hold on
str1='o<>sx+';str2='krgbc';
for i=1:length(position)
    plot3(position(i,1),position(i,2),position(i,3),[str2(mod(i,5)+1) str1(mod(i,6)+1)],'MarkerSize',10);
    vv=pos_good_dipoles(find(I==i),:);
    plot3(vv(:,1),vv(:,2),vv(:,3),[str2(mod(i,5)+1) str1(mod(i,6)+1)],'MarkerSize',2);
end
figure;
str1='oooooo';str2='kkkkk';
surf(xx,yy,zz,ones(size(zz)),'EdgeColor','none');alpha(.5);hold on
for i=1:length(position)
    plot3(position(i,1),position(i,2),position(i,3),[str2(mod(i,5)+1) str1(mod(i,6)+1)],'MarkerSize',10,'MarkerFaceColor','k');
    vv=pos_good_dipoles(find(I==i),:);
    plot3(vv(:,1),vv(:,2),vv(:,3),[str2(mod(i,5)+1) str1(mod(i,6)+1)],'MarkerSize',2,'MarkerFaceColor','auto');
end
% Stima delle attività dei dipoli che hanno prodotto il segnale EEG
% registrato
Nsource=3*sum(DipoleField.inside);
good_dipoles=find(DipoleField.inside==1);
A = zeros(Nch, Nsource);
n = 1;
for i=1:Nsource/3
    A(:,n:n+2) = DipoleField.leadfield{good_dipoles(i)}(good,:);
    n = n + 3;
end

% stima delle attività dei dipoli, minimizzando lo scarto quadratico con un
% termine di regolarizzazione che penalizza l'energia 
L=eig(A*A');Lambda=max(L)/1000;
denom = (A*A'+Lambda*eye(size(A*A')));
w = A' / denom;

% stima del segnale
est=(A*w)*record;
% Errore nella ricostruzione del segnale come somma delle attività dei
% dipoli stimati
MSE=(std(record-est)/std(record))^2;
disp(['MSE = ' num2str(MSE*100) '%'])
figure;
hl(1)=subplot(211);
mostra_segnali(record/6/std(record(:)),'k','',fs);mostra_segnali(est/6/std(record(:)),'r','',fs);title('EEG signal')
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