function [truePositives, trueNegatives, falsePositives, falseNegatives] = calculateConfusionMatrix(image1, image2)
    % Convert images to binary if not already
    image1 = im2bw(image1);
%   image2 = im2bw(image2);

    % Calculate true positives, true negatives, false positives, false negatives
    truePositives = sum(image1(:) & image2(:));
    trueNegatives = sum(~image1(:) & ~image2(:));
    falsePositives = sum(~image1(:) & image2(:));
    falseNegatives = sum(image1(:) & ~image2(:));
    
    
     confMatrix = confusionmat(image2(:),image1(:));
     disp('Confusion Matrix:');
     disp(confMatrix);

   
end
   