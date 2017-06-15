function scores = getScores(counter,ITER,ITER2,SNR,B,prm)
    clc;
    close all;
    
    testDir = '';
    resDir = sprintf('%sres/res%d_%d/',testDir,ITER,counter);
    
    if exist(sprintf('%sscores/s%d_%d_%d.mat',testDir,ITER,ITER2,counter),'file')
        return;
    end
    
    addpath('Fmeasure')
    
    f = zeros(SNR,100);    
    for i = 1:SNR
        R = im2double(imread(sprintf('%s%d.png',resDir,i)));
        [fcur,thresh] = evaluation_bdry_image(R,B,100);
        f(i,:) = fcur;        
    end

    f = mean(f);
    T = thresh(find(f == max(f),1));
    d = 0.00;
    Fopt = 0;

    for t = max(0,T-d):0.01:min(1,T+d)
        disp(t);
        f = zeros(1,SNR);

        for i = 1:SNR
            R = im2double(imread(sprintf('%s%d.png',resDir,i)));
            fcur = evaluation_bdry_image(R>=T,B,1);
            f(1,i) = fcur;
        end


        F = mean(f);

        if F>=Fopt
            Fopt = F;
            Topt = t;
            optScores = f;
        end
    end


    for i = 1:SNR
            R = im2double(imread(sprintf('%s%d.png',resDir,i)));
            binRes = R>=Topt;
            imwrite(binRes,sprintf('%sB%d.png',resDir,i),'PNG');
    end
    s.scores = optScores;
    s.prm = prm;
    disp(optScores);
    disp(Fopt);
    s.T = Topt;
    s.F = Fopt;
    mkdir(sprintf('%sscores/',testDir));
    save(sprintf('%sscores/s%d_%d_%d.mat',testDir,ITER,ITER2,counter),'-struct','s');
end