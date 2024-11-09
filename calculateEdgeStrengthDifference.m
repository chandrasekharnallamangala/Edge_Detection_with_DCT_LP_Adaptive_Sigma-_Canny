function edgeStrengthDifference = calculateEdgeStrengthDifference(image1, image2)
    % Convert images to grayscale if they are color images
    if size(image1, 3) == 3
        image1 = rgb2gray(image1);
    end

    if size(image2, 3) == 3
        image2 = rgb2gray(image2);
    end

    % Apply the Sobel operator to calculate gradient magnitude for each image
    sobelFilter = fspecial('sobel');

    gradientX1 = imfilter(double(image1), sobelFilter, 'conv');
    gradientY1 = imfilter(double(image1), sobelFilter', 'conv');
    gradientMagnitude1 = sqrt(gradientX1.^2 + gradientY1.^2);

    gradientX2 = imfilter(double(image2), sobelFilter, 'conv');
    gradientY2 = imfilter(double(image2), sobelFilter', 'conv');
    gradientMagnitude2 = sqrt(gradientX2.^2 + gradientY2.^2);

    % Calculate the absolute difference in gradient magnitude
    edgeStrengthDifference = abs(gradientMagnitude1 - gradientMagnitude2);
end
