% Read the image
input_image = imread('37B.gif'); % Replace with your image file path

% Convert the image to grayscale if it's a color image
if size(input_image, 3) == 3
    gray_image = rgb2gray(input_image);
else
    gray_image = input_image;
end

% Define sigma value for the LoG filter
sigma_value = 1.4;

% Apply Laplacian of Gaussian (LoG) filter
log_filter = fspecial('log', round(6 * sigma_value + 1), sigma_value);
filtered_image = imfilter(double(gray_image), log_filter, 'symmetric', 'conv');

% Find zero-crossings in the LoG filtered image
edge_image = edge(filtered_image, 'zerocross');

% Display the original and edge-detected images
figure;
subplot(1, 2, 1);
imshow(uint8(gray_image));
title('Original Image');

subplot(1, 2, 2);
imshow(edge_image);
title('LoG Edge Detection');
