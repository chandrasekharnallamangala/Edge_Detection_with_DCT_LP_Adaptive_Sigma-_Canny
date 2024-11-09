% Define a custom edge detection filter (example: simple horizontal edge filter)
customFilter = [-1 -1 -1; 2 2 2; -1 -1 -1]; % Adjust as needed

% Convolve the image with the custom filter
filteredImage = conv2(double('source06_3.tif'), customFilter, 'same');
edges_custom = abs(filteredImage) > threshold_value; % Define a threshold for edge detection

% Display the custom edge-detected image
figure;
imshow(edges_custom);
title('Custom Edge Detection');
