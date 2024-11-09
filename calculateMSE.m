function mseValue = calculateMSE(originalImage, noisyImage)
    % Ensure both images are of the same size
    assert(all(size(originalImage) == size(noisyImage)), 'Images must be of the same size.');

    % Convert images to double to avoid integer arithmetic issues
    originalImage = double(originalImage);
    noisyImage = double(noisyImage);

    % Calculate mean squared error (MSE)
    mseValue = mean((originalImage(:) - noisyImage(:)).^2);
end
