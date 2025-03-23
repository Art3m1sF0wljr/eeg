function mostra_segnali(signal,str,tit,fsamp)
[numChannels,len]=size(signal);t=1/fsamp:1/fsamp:len/fsamp;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%for i=1:numChannels
    %signal(i,:)=signal(i,:)-mean(signal(i,:));
%end
for i=1:numChannels
    vv=signal(i,:);
    plot(t,vv+i,'Color',str)
    hold on
end
title(tit)
