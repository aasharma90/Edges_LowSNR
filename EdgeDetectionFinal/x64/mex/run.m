function E = run( I,prm )
    %clc;
    close all;
    
    
    I = im2double(I);
    E = runIm(I,prm);
    
    if prm.addShift
        Inew = zeros(size(I));
        Is = I(3:end,3:end);
        Inew(1:end-2,1:end-2) = Is;
        Inew(end-1:end,1:end) = I(1:2,1:end);
        Inew(1:end,end-1:end) = I(1:end,1:2);
        Enew = runIm(Inew,prm);
        E(3:end,3:end) = max(E(3:end,3:end),Enew(1:end-2,1:end-2));
    end
end

