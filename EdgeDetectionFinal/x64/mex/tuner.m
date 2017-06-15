function tuner()
    clc;
    clear;
    close all;

    minSnr = 0.0;
    SNR = 21;
    I = im2double(imread('curve.png'));
    groundTruth = edge(I,'canny');
    gt = {groundTruth};
    prm = getPrm();
    prm.minSnr = minSnr;
    tuneEdges(1,SNR,gt,prm);
    p = sum(groundTruth(:))/numel(groundTruth);
end


function tuneEdges(ITER,SNR,gt,prm)
    prm.algo = 'Edges';
    for ITER2 = 1:100
        mkdir(sprintf('4test/params%d',ITER));
        tune(ITER2,prm,ITER,ITER2,SNR,gt);
    end
end

function tune(counter,prm,ITER,ITER2,SNR,gt)
    s.prm = prm;
    disp(prm);
    disp(counter);
    
    curMat = sprintf('4test/params%d/%d.mat',ITER,counter);
    if ~exist(curMat,'file')
        save(curMat,'-struct','s');
        for index = 1:SNR
            runWrapper(index,counter,ITER);
        end
    end
    %getScores(counter,ITER,ITER2,SNR,gt,prm);
    %if exist(sprintf('4test/res/res%d_%d/',ITER,counter),'dir');
    %    rmdir(sprintf('4test/res/res%d_%d/',ITER,counter),'s');
    %end
end

function mergeScores(rounds,algIndex,SNR)
    ITER2 = rounds;

    for alg=algIndex
        scoresAll = zeros(rounds,SNR);
        tAll = zeros(rounds,1);
        try
        for iter = 1:ITER2
            load(sprintf('4test/scores/s%d_%d_1.mat',alg,iter));
            scoresAll(iter,:) = scores;
            tAll(iter,1) = T;
        end
        catch MException
            continue;
        end
        if ITER2 > 1
            s.scores = median(scoresAll);
            s.T = mean(tAll);
            s.F = median(mean(scoresAll,2));
        else
            s.scores = scoresAll;
            s.T = tAll;
            s.F = mean(scoresAll);
        end
        save(sprintf('4test/scores/s%d_1.mat',alg),'-struct','s');
    end
end
