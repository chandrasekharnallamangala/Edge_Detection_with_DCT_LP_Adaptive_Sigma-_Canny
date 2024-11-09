
% function groundTruthImage = ProposedSfunc2(inputImage)
%     % Read the input image
%     I = inputImage;
% 
%     % Convert to grayscale if it's a color image
%     if size(I, 3) == 3
%         J = rgb2gray(I);
%     else
%         J = I;
%     end
%     
%     % Preprocess the image to enhance edges
%     J = imsharpen(J); % Sharpen the image
%     J = medfilt2(J, [3 3]); % Apply median filtering to further reduce noise
% 
%     % Calculate the gradient magnitude using Sobel
%     gradient_x = edge(J, 'Sobel', 'horizontal');
%     gradient_y = edge(J, 'Sobel', 'vertical');
%     gradient_magnitude = sqrt(gradient_x.^2 + gradient_y.^2);
% 
%     % Calculate the mean and standard deviation of the gradient magnitude
%     mean_gradient = mean(gradient_magnitude(:));
%     std_gradient = std(gradient_magnitude(:));
% 
%     % Calculate adaptive thresholds
%     Ts_high = mean_gradient + 0.8 * std_gradient; % Increase multiplier for higher sensitivity 0.8,0.5  + 0.8
%     Ts_low = 0.6 * Ts_high; % Adjust ratio for lower threshold 0.6
% 
%     % Adjust sigma value based on image size and structure
%     sig = * (size(J, 1) / 512); % Scale sigma with image size
% 
%     % Apply Canny edge detection
%     Cs = edge(J, 'canny', [Ts_low Ts_high], sig);
% 
%     % Return the ground truth binary image
%     groundTruthImage = Cs;
%     
%     % Display the result (optional)
% %     imshow(Cs);
% end


function groundTruthImage = ProposedSfunc2(image)
    % Read the input image
    img = image;
    
    % Convert image to grayscale if it's not already
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    % Set sigma value based on image size 650
    sigma = min(size(img)) / 250;
    
    % Apply Canny edge detection without thresholding
    edges = edge(img, 'Canny', [], sigma);

    groundTruthImage = edges;
end







































% 
% function groundTruthImage = ProposedSfunc2(inputImage)
%     % Read the input image
%     I = inputImage;
% 
%     % Convert to grayscale if it's a color image
%     if size(I, 3) == 3
%         J = rgb2gray(I);
%     else
%         J = I;
%     end
%     
%     % Preprocess the image to enhance edges
%     J = imsharpen(J); % Sharpen the image
%     J = medfilt2(J, [3 3]); % Apply median filtering to further reduce noise
% 
%     % Calculate the gradient magnitude using Sobel
%     gradient_x = edge(J, 'Sobel', 'horizontal');
%     gradient_y = edge(J, 'Sobel', 'vertical');
%     gradient_magnitude = sqrt(gradient_x.^2 + gradient_y.^2);
% 
%     % Calculate the mean and standard deviation of the gradient magnitude
%     mean_gradient = mean(gradient_magnitude(:));
%     std_gradient = std(gradient_magnitude(:));
% 
%     % Calculate adaptive thresholds
%     Ts_high = mean_gradient + 0.8 * std_gradient; % Increase multiplier for higher sensitivity 0.8,0.5
%     Ts_low = 0.6 * Ts_high; % Adjust ratio for lower threshold 0.6
% 
%     % Adjust sigma value based on image size and structure
%     sig = 3.14159 * (size(J, 1) / 512); % Scale sigma with image size
% 
%     % Apply Canny edge detection
%     Cs = edge(J, 'canny', [Ts_low Ts_high], sig);
% 
%     % Return the ground truth binary image
%     groundTruthImage = Cs;
%     
%     % Display the result (optional)
% %     imshow(Cs);
% end





% function groundTruthImage = ProposedSfunc2(inputImage)
%     % Read the input image
%     I = inputImage;
% 
%     % Convert to grayscale if it's a color image
%     if size(I, 3) == 3
%         J = rgb2gray(I);
%     else
%         J = I;
%     end
%     
%     % Preprocess the image to enhance edges
%     J = imsharpen(J); % Sharpen the image
%     J = medfilt2(J, [3 3]); % Apply median filtering to further reduce noise
% 
%     % Calculate the gradient magnitude using Sobel
%     gradient_x = edge(J, 'Sobel', 'horizontal');
%     gradient_y = edge(J, 'Sobel', 'vertical');
%     gradient_magnitude = sqrt(gradient_x.^2 + gradient_y.^2);
% 
%     % Calculate the mean and standard deviation of the gradient magnitude
%     mean_gradient = mean(gradient_magnitude(:));
%     std_gradient = std(gradient_magnitude(:));
% 
%     % Calculate adaptive thresholds
%     Ts_high = mean_gradient + 0.8 * std_gradient; % Increase multiplier for higher sensitivity
%     Ts_low = 0.6 * Ts_high; % Adjust ratio for lower threshold
% 
%     % Adjust sigma value based on image size and structure
%     sig = 3.14159 * (size(J, 1) / 512); % Scale sigma with image size
%     
%     % Alternatively, you can adapt sigma based on the image gradient magnitude properties
%     % sig = 3.14159 * (size(J, 1) / 512) * (mean_gradient / 100); % Scale sigma with image size and mean gradient
% 
%     % Apply Canny edge detection
%     Cs = edge(J, 'canny', [Ts_low Ts_high], sig);
% 
%     % Return the ground truth binary image
%     groundTruthImage = Cs;
%     
%     % Display the result (optional)
% %     imshow(Cs);
% end

% 
% function groundTruthImage = ProposedSfunc2(inputImage)
%     % Read the input image
%     I = inputImage;
% 
%     % Convert to grayscale if it's a color image
%     if size(I, 3) == 3
%         J = rgb2gray(I);
%     else
%         J = I;
%     end
%     
%     % Preprocess the image to enhance edges
%     J = imsharpen(J); % Sharpen the image
%     J = medfilt2(J, [3 3]); % Apply median filtering to further reduce noise
% 
%     % Calculate the gradient magnitude using Sobel
%     gradient_x = edge(J, 'Sobel', 'horizontal');
%     gradient_y = edge(J, 'Sobel', 'vertical');
%     gradient_magnitude = sqrt(gradient_x.^2 + gradient_y.^2);
% 
%     % Calculate local mean and standard deviation of the gradient magnitude
%     local_mean = mean2(gradient_magnitude);
%     local_std = std2(gradient_magnitude);
% 
%     % Calculate adaptive thresholds
%     Ts_multiplier_high = 0.8; % Multiplier for higher sensitivity
%     Ts_ratio_low = 0.6; % Ratio for lower threshold
%     Ts_high = local_mean + Ts_multiplier_high * local_std; % Increase multiplier for higher sensitivity
%     Ts_low = Ts_ratio_low * Ts_high; % Adjust ratio for lower threshold
% 
%     % Calculate sigma dynamically based on image size and structure
%     sig_scale_factor = 0.5; % Scale factor for sigma adjustment
%     sig = pi * (size(J, 1) / 512) * sig_scale_factor; % Scale sigma with image size
% 
%     % Apply Canny edge detection
%     Cs = edge(J, 'canny', [Ts_low Ts_high], sig);
% 
%     % Return the ground truth binary image
%     groundTruthImage = Cs;
%     
%     % Display the result (optional)
% %     imshow(Cs);
% end
% 
