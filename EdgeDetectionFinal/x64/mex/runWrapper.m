function runWrapper(imgInd,prmInd,ITER)
    curDir = '';
    load(sprintf('4test/%sparams%d/%d.mat',curDir,ITER,prmInd)); 
    I = im2double(imread('curve.png'));
    sigma = 0.1;
    snr = (imgInd-1)*0.2+prm.minSnr;
    I = I./max(I(:));
    I = I>0.5;
    rand('seed',sum(100*clock));
    clip = @(x)min(max(0,x),1);
    I = clip(I*snr*sigma+0.5+sigma*randn(size(I)));
    rand('seed',sum(100*clock));
    I = imnoise(I,'salt & pepper',0.01);
    I = clip(I);
    
    if strcmp(prm.algo,'Edges')
        E = run(I,prm);
        if prm.linearStrech
            m = max(E(:));
            E = E./(m + (m==0));
        end
    end
    d1 = sprintf('res');
    if ~exist(d1,'dir')
        mkdir(d1);
    end
    d2 = sprintf('res/res%d_%d',ITER,prmInd);
    if ~exist(d2,'dir')
        mkdir(d2);
    end
    imwrite(E,sprintf('res/res%d_%d/%d.png',ITER,prmInd,imgInd));
    imwrite(I,sprintf('res/res%d_%d/%dI.png',ITER,prmInd,imgInd));
end