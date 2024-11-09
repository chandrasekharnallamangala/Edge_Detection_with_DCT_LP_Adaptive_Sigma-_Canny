function EvaluateParams(im2, resultImage,ns,x)
% MSE Calculation

   % Ensure both images are of the same size
    assert(all(size(im2) == size(resultImage)), 'Images must be of the same size.');

    % Convert images to double to avoid integer arithmetic issues
    originalImage = double(im2);
    noisyImage = double(resultImage);

    % Calculate mean squared error (MSE)
    mseValue = mean((originalImage(:) - noisyImage(:)).^2);

   
    fprintf('MSE: %.2f dB\n', mseValue);


%RMSE calculation
    rmse = sqrt(mseValue);
    disp(['RMSE: ' ,num2str(rmse)]);


%Edge strength calculation
    es = 1 /mseValue;
    disp(['Edge strength:   ',num2str(es)]);


%PSNR calculation
 if size(im2,3)~=1     
   org=rgb2ycbcr(im2);  
   test=rgb2ycbcr(resultImage);  
   Y1=org(:,:,1);  
   Y2=test(:,:,1);  
   Y1=double(Y1);   
   Y2=double(Y2);  
 else            
     Y1=double(im2);  
     Y2=double(resultImage);  
 end  
   
% if nargin < 2      
%    D = Y1;  
% else  
%   if any(size(Y1)~=size(Y2))  
%     error('The input size is not equal to each other!');  
%   end  
 D = Y1 - Y2;   
% end  
MSE = sum(D(:).*D(:)) / numel(Y1);   
PSNR = 10*log10(255^2 / MSE); 
fprintf('PSNR: %.2f dB\n', PSNR);


%Accuracy calculation
assert(all(size(im2) == size(resultImage)), 'Images must be of the same size.');

    % Convert images to double to avoid integer arithmetic issues
    originalImage = double(im2);
    processedImage = double(resultImage);

    % Count matching pixels
    matchingPixels = sum(originalImage(:) == processedImage(:));

    % Calculate accuracy as the percentage of matching pixels
    totalPixels = numel(originalImage);
    accuracy = (matchingPixels / totalPixels) * 100;
    fprintf('Accuracy: %.2f%%\n', accuracy);



% Confusion matrix calculation
% Convert images to binary if not already
    image1 = im2bw(im2);
   image2 = im2bw(resultImage);

    % Calculate true positives, true negatives, false positives, false negatives
    truePositives = sum(image1(:) & image2(:));
    trueNegatives = sum(~image1(:) & ~image2(:));
    falsePositives = sum(~image1(:) & image2(:));
    falseNegatives = sum(image1(:) & ~image2(:));
    disp(['True Positives: ' num2str(truePositives)]);
disp(['True Negatives: ' num2str(trueNegatives)]);
disp(['False Positives: ' num2str(falsePositives)]);
disp(['False Negatives: ' num2str(falseNegatives)]);

 confMatrix = confusionmat(image1(:),image2(:));
     disp('Confusion Matrix:');
     disp(confMatrix);


%accuracy
acc = ( truePositives + trueNegatives )/ (truePositives + falsePositives + trueNegatives +falseNegatives);
disp(['Accuracy 2  : ' num2str(acc)]);

% Calculate precision
precision = truePositives / (truePositives + falsePositives);
disp(['Precision: ' num2str(precision)]);

% calculate Recall
recall = truePositives / (truePositives + falseNegatives);
disp(['Recall: ' num2str(recall)]);

% calculate FPR
fpr = falsePositives / (falsePositives + trueNegatives);
disp(['FPR: ' num2str(fpr)]);


% f1 score calculation
f1Score = 2 * (precision * recall) / (precision + recall);
disp(['F1 Score: ' num2str(f1Score)]);


%Euclidean distance calculation
euclideanDistance = calculateEuclideanDistance(im2,resultImage);
disp(['Euclidean Distance: ' num2str(euclideanDistance)]);









    








% export data to excel
columnname = {'NoiseLevel','MSError','R.M.S.E','Edge strength','P.S.N.R','Accuracy','TruePositive','TrueNegative','FalsePositive','FalseNegative','Accuracy2','precesion','Recall','FalsePositiveRate','F1Score','EuclideanDistance'};
resultData = table(ns(:),mseValue(:),rmse(:),es(:),PSNR(:),accuracy(:),truePositives(:),trueNegatives(:), ...
     falsePositives(:),falseNegatives(:),acc(:), precision(:),recall(:),fpr(:),f1Score(:),euclideanDistance(:),'VariableNames',columnname);

% Determine the sheet number based on the value of x
if x == 1
    sheetNumber = 1;
elseif x == 2
    sheetNumber = 2;
elseif x == 3
    sheetNumber = 3;
else
    sheetNumber = 4;
end

% Append data to the specified sheet in the Excel file
excelFilename = 'parameters.xlsx';
writetable(resultData, excelFilename, 'WriteMode', 'append', 'Sheet', sheetNumber);










% excelFilename = 'parameters.xlsx';
%  writetable(resultData,excelFilename,'WriteMode','append');




end