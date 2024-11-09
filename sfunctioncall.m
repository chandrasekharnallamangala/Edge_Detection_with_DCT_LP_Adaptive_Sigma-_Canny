clear all;
close all;
inputImage = imread("cameraman.tif");

groundTruthImage = edgeDetectionWithSFunction(inputImage);
