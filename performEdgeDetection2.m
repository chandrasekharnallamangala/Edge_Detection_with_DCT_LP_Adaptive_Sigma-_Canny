function edgeStrength = performEdgeDetection2(inputImage)
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

    % Perform edge detection only on scale 1
    edges = edge(pyramid{1}, 'Canny'); % Apply Canny edge detector on scale 1
    
    % Compute gradient magnitude for numeric values of edge strengths
    [Gx, Gy] = gradient(double(pyramid{1}));
    edgeStrength = sqrt(Gx.^2 + Gy.^2);

    % Display the original image, edge strength map, and the result of edge detection
    figure;

    subplot(1, 3, 1);
    imshow(originalImage);
    title('Original Image');

    subplot(1, 3, 2);
    imshow(edgeStrength, []);
    title('Edge Strength Map (Numeric Values)');

    subplot(1, 3, 3);
    imshow(inputImage);
    hold on;
    visboundaries(edges, 'Color', 'r');
    title('Edge Detection Result');

    colormap('gray');
end

