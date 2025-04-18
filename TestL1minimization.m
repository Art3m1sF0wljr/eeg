clear all;close all
load('DipoleField');Nsources=3*sum(DipoleField.inside);
% Creo una superficie che interpoli le posizioni degli elettrodi
S = scatteredInterpolant(elec.elecpos(:,1),elec.elecpos(:,2),elec.elecpos(:,3),'natural');%come in estimate_sources
% Aggiungo punti lungo le direzioni x e y, usando S per valutare il valore
% di z corrispondente sullo scalpo
minX=min(elec.elecpos(:,1));
maxX=max(elec.elecpos(:,1));
minY=min(elec.elecpos(:,2));
maxY=max(elec.elecpos(:,2));
[xx,yy]=meshgrid(linspace(minX,maxX,20),linspace(minY,maxY,20));
zz=S(xx,yy);
% controllo quali punti (x,y) sono dentro al convex hull della testa
tess = convhulln(elec.elecpos(:,1:2));
in = inhull([xx(:) yy(:)],elec.elecpos(:,1:2),tess);
bad=find(in==0);
zz(bad)=nan;
%fino a qui come in estimate_sources

% Simulo l'attivazione di alcune sorgenti
good_dipoles=find(DipoleField.inside==1);%only those inside the scalp
Nch=length(DipoleField.label);
A = zeros(Nch, Nsources); %leadfield matrix
n = 1;
for i=1:Nsources/3
    A(:,n:n+2) = DipoleField.leadfield{good_dipoles(i)};
    n = n + 3;
end
% scelgo un certo numero di sorgenti distribuite in modo random
active_sources=5;
Xsimulated=[rand(active_sources,1);%Xsimulated is a vector representing the true source activity
zeros(Nsources-active_sources,1)];%The remaining dipoles are inactive (zero).
AS=randperm(Nsources);%The vector is randomly shuffled using randperm, so the active sources are randomly distributed.
Xsimulated=Xsimulated(AS);
b=A*Xsimulated;%represents the simulated EEG measurements at the electrodes
%It is computed by multiplying the lead field matrix A with the simulated source activity Xsimulated.
%This step mimics the forward problem in EEG source modeling, where scalp potentials are generated from brain sources
F = scatteredInterpolant(elec.elecpos(:,1),elec.elecpos(:,2),elec.elecpos(:,3),b,'natural');
surf(xx,yy,zz,F(xx,yy,zz),'EdgeColor','none');
alpha(.6);
xlabel('x');
ylabel('y');
zlabel('z');
axis equal;
hold on;
qq=find(Xsimulated>0);%Finds the indices of the active sources (qq).
for Num=1:length(qq)
    meanA=Xsimulated(qq(Num));%Determines how strongly each source is activated (meanA)
	Num_dipole=round(qq(Num)/3+1);%Maps the dipole index qq(Num) to the corresponding position in 3D space (DipoleField.pos).
	color_level=round(100*(.9-.9*meanA/max(Xsimulated)))/100;
    plot3(DipoleField.pos(good_dipoles(Num_dipole),1),DipoleField.pos(good_dipoles(Num_dipole),2),DipoleField.pos(good_dipoles(Num_dipole),3),'o','color',color_level*[1 1 1],'MarkerSize',1+5*meanA/max(Xsimulated),'LineWidth',1+2*meanA/max(Xsimulated))
end


iter=100;%Number of iterations
lambda=max(eig(A*A'))/1000;
[X,L2norm,L1norm] = underdeterminedL1minimization(b,A,iter,lambda);
%praticamente genera un eeg signal partendo dall'attivazione di alcune sorgenti
%simulando un eeg
%poi tramite underdeterminedL1minimization() &normalisation tipo in estimate_sources
%Recover the original source activations Xsimulated from the measured EEG b.
%L1 minimization is used because it promotes sparse solutions (only a few active sources).


figure;
subplot(2,1,1);plot(Xsimulated,'k');hold on;plot(X(:,round(end/5)),'r');plot(X(:,round(end/2)),'m');plot(X(:,end),'g');legend('simulated',[num2str(iter/5) ' iter'],[num2str(iter/2) ' iter'],[num2str(iter) ' iter'])
subplot(2,1,2);plot(L2norm);hold on;plot(L1norm);xlabel('iterations');legend('L_2 norm','L_1 norm')
str{1}='MSE solution';
str{2}='L1 minimization';
cont=0;
for ii=[1 iter]
    cont=cont+1;
    figure
    vet=X(:,ii);
    F = scatteredInterpolant(elec.elecpos(:,1),elec.elecpos(:,2),elec.elecpos(:,3),A*vet,'natural');
    surf(xx,yy,zz,F(xx,yy,zz),'EdgeColor','none');alpha(.6);xlabel('x');ylabel('y');zlabel('z');axis equal;hold on;
    qq=find(vet>0);
    for Num=1:length(qq)
        meanA=X(qq(Num));Num_dipole=ceil(qq(Num)/3);color_level=round(100*(.9-.9*meanA/max(Xsimulated)))/100;
        plot3(DipoleField.pos(good_dipoles(Num_dipole),1),DipoleField.pos(good_dipoles(Num_dipole),2),DipoleField.pos(good_dipoles(Num_dipole),3),'o','color',color_level*[1 1 1],'MarkerSize',1+5*meanA/max(Xsimulated),'LineWidth',1+2*meanA/max(Xsimulated))
    end
    title(str{cont})
end


% simulazione di un segnale variabile nel tempo
fs=64;tt=-.5:1/fs:.5;ss=.02;g=diff(exp(-tt.^2/2/ss^2));
len_epoch=length(tt);
nn=3;vv=[ones(nn,1);zeros(len_epoch-nn,1)];
Xsim=zeros(Nsources,len_epoch);
qq=find(Xsimulated>0);
for i=1:length(qq)
   Xsim(qq(i),:)=Xsimulated(qq(i))*conv(vv(randperm(len_epoch)),g,'same'); 
end
b=A*Xsim;
[X,L2norm,L1norm] = underdeterminedL1minimization_epoch(b,A,iter,lambda);
%the previous:
%It plots the true source activity (Xsimulated, black) against the estimated sources at different iterations
%Early iterations (iter/5, red)
%Mid iterations (iter/2, magenta)
%Final solution (iter, green)
%L2 norm → Measures the squared error (total energy).
%L1 norm → Measures sparsity of the solution.
%L1 norm should decrease, enforcing a sparse solution.
%L2 norm should also decrease, showing an improvement in the fit.
figure
h(1)=subplot(221);
F = scatteredInterpolant(elec.elecpos(:,1),elec.elecpos(:,2),elec.elecpos(:,3),mean(abs(b),2),'natural');
surf(xx,yy,zz,F(xx,yy,zz),'EdgeColor','none');alpha(.6);xlabel('x');ylabel('y');zlabel('z');axis equal;hold on;
qq=find(Xsimulated>0);
for Num=1:length(qq)
    meanA=Xsimulated(qq(Num));Num_dipole=round(qq(Num)/3+1);color_level=round(100*(.9-.9*meanA/max(Xsimulated)))/100;
    plot3(DipoleField.pos(good_dipoles(Num_dipole),1),DipoleField.pos(good_dipoles(Num_dipole),2),DipoleField.pos(good_dipoles(Num_dipole),3),'o','color',color_level*[1 1 1],'MarkerSize',1+5*meanA/max(Xsimulated),'LineWidth',1+2*meanA/max(Xsimulated))
end
h(2)=subplot(222);
F = scatteredInterpolant(elec.elecpos(:,1),elec.elecpos(:,2),elec.elecpos(:,3),mean(abs(b),2),'natural');
surf(xx,yy,zz,F(xx,yy,zz),'EdgeColor','none');alpha(.6);xlabel('x');ylabel('y');zlabel('z');axis equal;hold on;
qq=find(sum(abs(X),2)>0);
for Num=1:length(qq)
    meanA=mean(abs(X(qq(Num),:)),2);Num_dipole=round(qq(Num)/3+1);color_level=round(100*(.9-.9*meanA/max(abs(X(:)))))/100;
    try;plot3(DipoleField.pos(good_dipoles(Num_dipole),1),DipoleField.pos(good_dipoles(Num_dipole),2),DipoleField.pos(good_dipoles(Num_dipole),3),'o','color',color_level*[1 1 1],'MarkerSize',1+5*meanA/max(Xsimulated),'LineWidth',1+2*meanA/max(abs(X(:))));end
end
hlink = linkprop(h,{'CameraPosition','CameraUpVector'}); 
subplot(212);mostra_segnali(b/max(b(:)),'k','',fs);set(gca,'YTick',[1:Nch]);set(gca,'YTickLabel',DipoleField.label)
mostra_segnali(A*X/max(b(:)),'r','',fs);
