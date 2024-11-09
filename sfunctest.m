% Read the input image

I = imread('10-lena-grey.png');
I2 = imread("10-lena.png");
[m, n, d] = size(I);
if d == 3
    J = rgb2gray(I);
else
    J = I;
end

%  sig =2.71828;
% default sigma value is 1
% sig = 1.38e-23;
% Ts_high = 1;
% Ts_low = 0.4 * Ts_high;
% sig = 6.626e-34;
 sig = 3.14159;
% sig = 1; 
Ts_high = 0.1;
Ts_low = 0.2 * Ts_high;
Thresh = [Ts_low Ts_high];
Cs = edge(J, 'canny', Thresh, sig);


%Accuracy calculation
assert(all(size(Cs) == size(I2)), 'Images must be of the same size.');

    % Convert images to double to avoid integer arithmetic issues
    originalImage = double(I2);
    processedImage = double(Cs);

    % Count matching pixels
    matchingPixels = sum(originalImage(:) == processedImage(:));

    % Calculate accuracy as the percentage of matching pixels
    totalPixels = numel(originalImage);
    accuracy = (matchingPixels / totalPixels) * 100;
    fprintf('Accuracy: %.2f%%\n', accuracy);

% Display the original and edged images side by side
figure;

subplot(1, 2, 1);
imshow(J);
title('Original Image');

subplot(1, 2, 2);
imshow(Cs);
title('Edge-Detected Image');

% % Return the ground truth binary image
% groundTruthImage = Cs;
