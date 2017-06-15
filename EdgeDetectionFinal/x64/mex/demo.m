clc;
clear;
close all;

tic;
prm = getPrm();
I = imread('Sqr.png');
E = runIm(I,prm);
figure;
subplot(1,2,1);
imshow(I);
subplot(1,2,2);
imshow(E, []);
toc;