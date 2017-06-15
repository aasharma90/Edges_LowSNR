clc;
clear all;
close all;

d = dir('src');
names = {d.name};
prm = getPrm2();
mkdir('res1_1');
for j=1:length(names)
    [~,name,ext] = fileparts(names{j});
    if strcmpi(ext,'.png')
        disp(name);
        I = im2double(imread(['src/' name ext]));
        E = run(I,prm);
        E = E*10;
        m = max(E(:));
        E = E./(m+(m==0));
        imwrite(E,['res1_1/' name ext]);
    end
end