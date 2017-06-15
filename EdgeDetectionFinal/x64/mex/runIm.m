function E = runIm(I,prm)
    I = im2double(I);
    E = EdgeDetection(I,prm.removeEpsilon,prm.maxTurn,prm.nmsFact,prm.splitPoints,prm.minContrast);
end