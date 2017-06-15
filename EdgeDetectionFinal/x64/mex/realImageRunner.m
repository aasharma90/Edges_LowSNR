clc;
clear all;
close all;
folder = 'C:\Users\yehonato\Dropbox\Phd\Research\EdgeDetection\ICCV Paper\Images\realImages2';
d = dir(folder);
names = {d.name};
names(1:2) = [];
names(end) = [];
prm = getPrm();
newFolder = [folder '/newRes2'];
mkdir(newFolder);

for j=1:length(names)
    disp(names{j});
    dst = [newFolder '/' names{j}];
    if exist(dst,'file')
        continue;
    end;
    I = im2double(imread([folder '/' names{j}]));
    if ndims(I) == 3
        I = rgb2gray(I);
    end
    I = I-min(I(:));
    I = I./max(I(:));
    I = imresize(I,0.5);
    E = runReal(I,prm);
    imwrite(E,dst);
end