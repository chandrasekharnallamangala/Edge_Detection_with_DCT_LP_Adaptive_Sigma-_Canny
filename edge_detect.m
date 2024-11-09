function outputImage = performEdgeDetection(inputImage)
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
    
    % Return the output image
    outputImage = edges;
end
