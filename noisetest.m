 % Read an image
originalImage = imread('10-lena-grey.png'); % Replace with your image file path
imshow(originalImage);
title('Original Image');

% Add salt and pepper noise
noisyImage = imnoise(originalImage, 'salt & pepper', 0); % Adjust noise density if needed
figure;
imshow(noisyImage);
title('Image with Salt and Pepper Noise');

% Apply a median filter to reduce the noise
filteredImage = medfilt2(noisyImage);
figure;
imshow(filteredImage);
title('Image after Median Filtering');

% Display side-by-side comparison
figure;
subplot(1,3,1), imshow(originalImage), title('Original Image');
subplot(1,3,2), imshow(noisyImage), title('Noisy Image');
subplot(1,3,3), imshow(filteredImage), title('Filtered Image');
