% Read an image (replace 'input_image.jpg' with your image filename)
inputImage = imread('f2.png');

% Convert to grayscale if needed
originalImage = im2gray(inputImage);

% Define parameters
numScales = 5; % Number of scales
scaleFactor = 0.5; % Scale factor for creating pyramid

% Create image pyramid
pyramid = cell(1, numScales);
pyramid{1} = originalImage;
for i = 2:numScales
    pyramid{i} = imresize(pyramid{i-1}, scaleFactor);
end

% Perform edge detection only on scale1
edges = edge(pyramid{1}, 'Canny'); % Apply Canny edge detector on scale1

% im=imwrite(edges);
% Display the original image and the edges
figure;
subplot(1, 2, 1);
imshow(originalImage);
title('Original Image');

subplot(1, 2, 2);
imshow(edges);
title('Edges (Canny on scale 1)');
