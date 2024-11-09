input_image = imread('coins.png'); % Replace with your image file path
sigma_value = 1.4; % Sigma value for LoG filter

log_edge_detection(input_image, sigma_value);
