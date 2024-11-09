% Read the input image
originalImage = imread('source06_3.tif');
originalImage = im2gray(originalImage); % Convert to grayscale if needed

% Apply Gaussian blur
blurredImage = imgaussfilt(originalImage, 2); % Adjust the standard deviation as needed

% Use adaptive thresholding (Otsu's method) for edge detection
threshold = graythresh(blurredImage);
edges = edge(blurredImage, 'Canny', threshold);

% Display original and edge-detected images
figure;
subplot(1, 2, 1);
imshow(originalImage);
title('Original Image');

subplot(1, 2, 2);
imshow(edges);
title('Adaptive Canny Edge Detection');
