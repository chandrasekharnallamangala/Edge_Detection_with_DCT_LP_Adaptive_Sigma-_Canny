function psnrValue = calculatePSNR(originalImage, noisyImage,mse)
    % Ensure both images are of the same size
    assert(all(size(originalImage) == size(noisyImage)), 'Images must be of the same size.');

    % Convert images to double to avoid integer arithmetic issues
    originalImage = double(originalImage);
    noisyImage = double(noisyImage);

    % Calculate mean squared error (MSE)
%     mse = mean((originalImage(:) - noisyImage(:)).^2);

    % Calculate PSNR
    maxValue = max(originalImage(:));
    psnrValue = 10 * log10(maxValue^2 / mse);
end
