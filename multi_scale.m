% Read the input image
originalImage = imread('256source04_3.tif');
originalImage = im2gray(originalImage); % Convert to grayscale if needed

% Define parameters
numScales = 5; % Number of scales
scaleFactor = 0.5; % Scale factor for creating pyramid

% Create image pyramid
pyramid = cell(1, numScales);
pyramid{1} = originalImage;
for i = 2:numScales
    pyramid{i} = imresize(pyramid{i-1}, scaleFactor);
end

% Initialize figure to display results
figure;
subplot(1, numScales + 1, 1);
imshow(originalImage);
title('Original Image');

% Perform edge detection on each scale
for i = 1:numScales
    edges = edge(pyramid{i}, 'Canny'); % Apply Canny edge detector
    subplot(1, numScales + 1, i + 1);
    imshow(edges);
    title(['Scale ', num2str(i)]);
end
