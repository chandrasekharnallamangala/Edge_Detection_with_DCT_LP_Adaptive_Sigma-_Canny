clear all
clc

% I=imread('256bookA.png');
[imt,pathname1] = double(uigetfile('*.*',''));figure(1);imshow(imt,[]);title('original image');ent_org=entropy(uint8(imt))
[im1,pathname2] = double(uigetfile('*.*',''));figure(2);imshow(im1,[]);title('multifocus image 1');
[im2,pathname3] = double(uigetfile('*.*',''));figure(3);imshow(im2,[]);title('multifocus image 2');

figure,imshow(filename1);
title('first image');

figure,imshow(filename2);
title('second image');



imt = double(imread('source06_3.tif'));figure(1);imshow(imt,[]);title('original image');ent_org=entropy(uint8(imt))

% Imaged to be fused
im1 = double(imread('source06_1.tif'));figure(2);imshow(im1,[]);title('multifocus image 1');
im2 = double(imread('source06_2.tif'));figure(3);imshow(im2,[]);title('multifocus image 2');