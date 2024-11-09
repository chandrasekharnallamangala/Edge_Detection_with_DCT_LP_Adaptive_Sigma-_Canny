function accuracy = calculateAccuracy(originalImage, processedImage)
    % Ensure both images are of the same size
    assert(all(size(originalImage) == size(processedImage)), 'Images must be of the same size.');

    % Convert images to double to avoid integer arithmetic issues
    originalImage = double(originalImage);
    processedImage = double(processedImage);

    % Count matching pixels
    matchingPixels = sum(originalImage(:) == processedImage(:));

    % Calculate accuracy as the percentage of matching pixels
    totalPixels = numel(originalImage);
    accuracy = (matchingPixels / totalPixels) * 100;
end
