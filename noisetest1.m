% Load the image
originalImage = imread('10-lena-grey.png'); % Replace 'your_image.jpg' with the actual image file

% Display the original image
figure;
subplot(1, 3, 1);
imshow(originalImage);
title('Original Image');

% Number of iterations
numIterations = 10;

% Range of noise density
minNoiseDensity = 0.01;
maxNoiseDensity = 0.1;

% Iterate through different noise densities
for iteration = 1:numIterations
    % Generate a random noise density in the specified range
    noiseDensity = (maxNoiseDensity - minNoiseDensity) * rand() + minNoiseDensity;
    
    % Add salt and pepper noise to the image
    noisyImage = imnoise(originalImage, 'salt & pepper', noiseDensity);
    
    % Apply median filter to remove salt and pepper noise
    filteredImage = medfilt2(noisyImage);
    
    % Display the noisy and filtered images for each iteration
    subplot(numIterations, 3, (iteration - 1) * 3 + 2);
    imshow(noisyImage);
    title(['Noisy Image (Density: ' num2str(noiseDensity) ')']);
    
    subplot(numIterations, 3, (iteration - 1) * 3 + 3);
    imshow(filteredImage);
    title('Filtered Image');
end

% Adjust the figure layout
sgtitle('Salt and Pepper Noise with Median Filter');
