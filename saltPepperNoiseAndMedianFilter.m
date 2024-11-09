function [filteredImages, noiseLevels] = saltPepperNoiseAndMedianFilter(inputImage)
    
    % Parameters
    iterations = 10;
    noiseLevels = linspace(0, 1, 10); % Generate a vector of 10 equally spaced values between 0 and 1



    
    filteredImages = cell(1, iterations);

    for i = 1:iterations
        % Add salt and pepper noise
        noisedImage = imnoise(inputImage, 'salt & pepper', noiseLevels(i));

        % Apply median filter
        filteredImage = medfilt2(noisedImage);
       
        % Store only the filtered image in the cell array
        filteredImages{i} = filteredImage;
    end
end
