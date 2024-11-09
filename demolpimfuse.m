%LP1D image fusion technique

clear all;
close all;
home;
L = 5; %No. of levels

% % Reference image
% [filename, pathname] = uigetfile({'*.tif;*.png;*.jpg;*.jpeg', 'Image Files (*.tif, *.png, *.jpg, *.jpeg)'}, 'Select the Reference Image');
% if isequal(filename, 0) || isequal(pathname, 0)
%     disp('User canceled the operation. Exiting.');
%     return;
% end
% imt = imread(fullfile(pathname, filename));
% imt = rgb2gray(imt); % Convert to grayscale if needed
% figure(1);
% imshow(imt, []);
% title('Original Image');
% ent_org = entropy(uint8(imt));
% 
% % Images to be fused
% [filename, pathname] = uigetfile({'*.tif;*.png;*.jpg;*.jpeg', 'Image Files (*.tif, *.png, *.jpg, *.jpeg)'}, 'Select Multifocus Image 1');
% if isequal(filename, 0) || isequal(pathname, 0)
%     disp('User canceled the operation. Exiting.');
%     return;
% end
% im1 = imread(fullfile(pathname, filename));
% im1 = rgb2gray(im1); % Convert to grayscale if needed
% figure(2);is it correct use the formula for edge strenth as inverse of mse


% imshow(im1, []);
% title('Multifocus Image 1');
% 
% [filename, pathname] = uigetfile({'*.tif;*.png;*.jpg;*.jpeg', 'Image Files (*.tif, *.png, *.jpg, *.jpeg)'}, 'Select Multifocus Image 2');
% if isequal(filename, 0) || isequal(pathname, 0)
%     disp('User canceled the operation. Exiting.');
%     return;
% end
% im2 = imread(fullfile(pathname, filename));
% im2 = rgb2gray(im2); % Convert to grayscale if needed
% figure(3);
% imshow(im2, []);
% title('Multifocus Image 2');

% 
% % 
% %  %Reference image
%  imt = double(imread('10-lena-grey.png'));figure(1);imshow(imt,[]);title('original image');
% % % 
% % % % Imaged to be fusedclockC
%  im1 = double(imread('10-lena-grey-Lft.png'));figure(2);imshow(im1,[]);title('multifocus image 1');
%  im4 = double(imread('10-lena-grey-Rt.png'));figure(2);imshow(im1,[]);title('multifocus image 2');




[imagename1,imagepath1]=uigetfile('images\*.jpg;*.bmp;*.png;*.tif;*.tiff;*.pgm;*.gif','Please choose the Original image');
imt=im2double(imread(strcat(imagepath1,imagename1)));

[imagename2, imagepath2]=uigetfile('images\*.jpg;*.bmp;*.png;*.tif;*.tiff;*.pgm;*.gif','Please choose the Left Blur image');
im1=im2double(imread(strcat(imagepath2,imagename2)));    
[imagename3, imagepath3]=uigetfile('images\*.jpg;*.bmp;*.png;*.tif;*.tiff;*.pgm;*.gif','Please choose the Right Blur image');
im4=im2double(imread(strcat(imagepath3,imagename3)));











%im3 = double(imread('toy3.gif'));figure(4);imshow(im3,[]);title('multifocus image 3');

% imt= imresize(imt,0.5);
% im1= imresize(im1,0.5);
% im2= imresize(im2,0.5);

%figure(2);
%subplot(121);imshow(im1,[]);title('multifocus image 1');
%subplot(122);imshow(im2,[]);title('multifocus image 2');


% Image fusion
imf = lpdctf(im1,im4,L);
imf1=cast(imf,'uint8');
imwrite(imf1,'lpdct_source06_3_L5.tif','tif');


 % Replace with your image file path
% sigma_value = 1.4; % Sigma value for LoG filter

%resultImage = log_edge_detection(imf, sigma_value);


% resultImage = performEdgeDetection(imf);
% im2 = performEdgeDetection(imt);






resultImage = edgeDetectionWithSFunction(imf);
[imagename1,imagepath3]=uigetfile('images\*.jpg;*.bmp;*.png;*.tif;*.tiff;*.pgm;*.gif','Please choose the Ground Truth image');
im2=im2double(imread(strcat(imagepath3,imagename1)));
% im2 = imread("10-lena.png");


% MAIN PROJECT
oi = edgeDetectionWithSFunction(imt);
 fi = ProposedSfunc2(imf);
 Params(im2,oi,1);
Params(im2,fi,2);



% Convert images to grayscale if they are RGB
if size(im2, 3) == 3
    im2 = rgb2gray(im2);
end
if size(oi, 3) == 3
    oi = rgb2gray(fi);
end

% Compute histograms of the images
hist1 = imhist(im2);
hist2 = imhist(fi);

% Normalize histograms
hist1 = hist1 / numel(im2);
hist2 = hist2 / numel(fi);

% Compute entropy
entropy1 = -sum(hist1(hist1 > 0) .* log2(hist1(hist1 > 0)));
entropy2 = -sum(hist2(hist2 > 0) .* log2(hist2(hist2 > 0)));

% Display entropy values
disp(['Entropy of image 1: ', num2str(entropy1)]);
disp(['Entropy of image 2: ', num2str(entropy2)]);

% Compute the absolute difference in entropy
entropy_diff = abs(entropy1 - entropy2);
disp(['Absolute difference in entropy: ', num2str(entropy_diff)]);




% Convert images to grayscale if they are RGB
if size(im2, 3) == 3
    im2 = rgb2gray(im2);
end
if size(oi, 3) == 3
    oi = rgb2gray(oi);
end

% Compute histograms of the images
hist1 = imhist(im2);
hist2 = imhist(oi);

% Normalize histograms
hist1 = hist1 / numel(im2);
hist2 = hist2 / numel(oi);

% Compute entropy
entropy1 = -sum(hist1(hist1 > 0) .* log2(hist1(hist1 > 0)));
entropy2 = -sum(hist2(hist2 > 0) .* log2(hist2(hist2 > 0)));

% Display entropy values
disp(['Entropy of image 1: ', num2str(entropy1)]);
disp(['Entropy of image 2: ', num2str(entropy2)]);

% Compute the absolute difference in entropy
entropy_diff = abs(entropy1 - entropy2);
disp(['Absolute difference in entropy: ', num2str(entropy_diff)]);















figure;
subplot(1, 2, 1);
imshow(oi);
title('Original Figure');

subplot(1, 2, 2);
imshow(fi);
title('fused Figure');
%END




% for iteration = 1:10
%     [filteredImage, noiseLevel] = saltPepperNoiseAndMedianFilter(imf);    
%     disp(['noise: ' ,num2str(noiseLevel)]);
%     fusenoisefilter = edgeDetectionWithSFunction(filteredImage);
% 
%  
% end



% % SALT & PEPPER NOISE WITH MEDIAN FILTER FOR FUSED IMAGE WITH EXISTING
[filteredImages, noiseLevels] = saltPepperNoiseAndMedianFilter(imf);

% Access filtered images for each iteration
for i = 1:length(filteredImages)
    disp(['Iteration ' num2str(i) ' - Noise Level: ' num2str(noiseLevels(i))]);

% Generate a filename based on the loop index
    outputFileName = fullfile('C:\Users\i\Desktop\FinalProject\DCTLP\DCTLP\op1', sprintf('output1b_image_%03d.png', i));
    % Save the output image
    imwrite(filteredImages{i}, outputFileName);



    fusenoisefilter = edgeDetectionWithSFunction(filteredImages{i});

    EvaluateParams(im2,fusenoisefilter,noiseLevels(i),1);
%     imshow(filteredImages{i});
    % Process or save the filtered images as needed
end



% % SALT & PEPPER NOISE WITH MEDIAN FILTER FOR ORIGINAL IMAGE WITH EXISTING

[filteredImages, noiseLevels] = saltPepperNoiseAndMedianFilter(imt);

% Access filtered images for each iteration
for i = 1:length(filteredImages)
    disp(['Iteration ' num2str(i) ' - Noise Level: ' num2str(noiseLevels(i))]);


% Generate a filename based on the loop index
    outputFileName = fullfile('C:\Users\i\Desktop\FinalProject\DCTLP\DCTLP\op2', sprintf('output2b_image_%03d.png', i));
    % Save the output image
    imwrite(filteredImages{i}, outputFileName);



    fusenoisefilter = edgeDetectionWithSFunction(filteredImages{i});

    EvaluateParams(im2,fusenoisefilter,noiseLevels(i),2);
%     imshow(filteredImages{i});
    % Process or save the filtered images as needed
end
% 
% 
% 
% 
% % SALT & PEPPER NOISE WITH MEDIAN FILTER FOR FUSED IMAGE WITH PROPOSED
[filteredImages, noiseLevels] = saltPepperNoiseAndMedianFilter(imf);

% Access filtered images for each iteration
for i = 1:length(filteredImages)
    disp(['Iteration ' num2str(i) ' - Noise Level: ' num2str(noiseLevels(i))]);




    % Generate a filename based on the loop index
    outputFileName = fullfile('C:\Users\i\Desktop\FinalProject\DCTLP\DCTLP\op3', sprintf('output3b_image_%03d.png', i));
    % Save the output image
    imwrite(filteredImages{i}, outputFileName);

%     Change edge detection function here
    fusenoisefilter = ProposedSfunc2(filteredImages{i});



    EvaluateParams(im2,fusenoisefilter,noiseLevels(i),3);
%     imshow(filteredImages{i});
    % Process or save the filtered images as needed
end



% SALT & PEPPER NOISE WITH MEDIAN FILTER FOR ORIGINAL IMAGE WITH PROPOSED
[filteredImages, noiseLevels] = saltPepperNoiseAndMedianFilter(imt);

% imshow(filteredImages{1});
%    title('filtered');

% Access filtered images for each iteration
for i = 1:length(filteredImages)
    disp(['Iteration ' num2str(i) ' - Noise Level: ' num2str(noiseLevels(i))]);
    
% 
%     imshow(filteredImages{1});
%     title('noised');

    % Generate a filename based on the loop index
    outputFileName = fullfile('C:\Users\i\Desktop\FinalProject\DCTLP\DCTLP\op4', sprintf('output4b_image_%03d.png', i));
    % Save the output image
    imwrite(filteredImages{i}, outputFileName);

     %     Change edge detection function here
    fusenoisefilter = ProposedSfunc2(filteredImages{i});  

    EvaluateParams(im2,fusenoisefilter,noiseLevels(i),4);
%     imshow(filteredImages{i});
    % Process or save the filtered images as needed
end



figure;

% Display images in a 2x3 grid
subplot(2, 3, 1);
imshow(imt);
title('Original Figure');

subplot(2, 3, 2);
imshow(im1);
title('multifocus image 1');

subplot(2, 3, 3);
imshow(im4);
title('multifocus image 2');

subplot(2, 3, 4);
imshow(imf);
title('Fused Image');

subplot(2, 3, 5);
imshow(im2);
title('Ground Truth');

subplot(2, 3, 6);
imshow(resultImage);
title('Existing code Image');
































% 
% %MSE calculation
% mse = calculateMSE(im2,resultImage);
% fprintf('MSE: %.2f dB\n', mse);
% 
% 
% %RMSE calculation
% rmse = sqrt(mse);
% disp(['RMSE: ' ,num2str(rmse)]);
% 
% %Edge strength calculation
% es = 1 /mse;
% disp(['Edge strength:   ',num2str(es)]);
% 
% %PSNR calculation
% psnrValue = psnr(im2,resultImage);
% fprintf('PSNR: %.2f dB\n', psnrValue);
% 
% %Accuracy calculation
% accuracyValue = calculateAccuracy(im2,resultImage);
% fprintf('Accuracy: %.2f%%\n', accuracyValue);
% 
% 
% % Confusion matrix calculation
% [truePositives, trueNegatives, falsePositives, falseNegatives] = calculateConfusionMatrix(im2,resultImage);
% disp(['True Positives: ' num2str(truePositives)]);
% disp(['True Negatives: ' num2str(trueNegatives)]);
% disp(['False Positives: ' num2str(falsePositives)]);
% disp(['False Negatives: ' num2str(falseNegatives)]);
% 
% %accuracy
% acc = ( truePositives + trueNegatives )/ (truePositives + falsePositives + trueNegatives +falseNegatives);
% disp(['Accuracy 2  : ' num2str(acc)]);
% 
% % Calculate precision
% precision = truePositives / (truePositives + falsePositives);
% disp(['Precision: ' num2str(precision)]);
% 
% % calculate Recall
% recall = truePositives / (truePositives + falseNegatives);
% disp(['Recall: ' num2str(recall)]);
% 
% % calculate FPR
% fpr = falsePositives / (falsePositives + trueNegatives);
% disp(['FPR: ' num2str(fpr)]);
% 
% 
% % f1 score calculation
% f1Score = 2 * (precision * recall) / (precision + recall);
% disp(['F1 Score: ' num2str(f1Score)]);
% 
% 
% %Euclidean distance calculation
% euclideanDistance = calculateEuclideanDistance(im2,resultImage);
% disp(['Euclidean Distance: ' num2str(euclideanDistance)]);








% 
% % Edge strength
% edgeStrengthDifference = calculateEdgeStrengthDifference(im2,resultImage);
% figure;
% subplot(1, 3, 1);
% imshow(image1);
% title('Image 1');
% 
% subplot(1, 3, 2);
% imshow(image2);
% title('Image 2');
% 
% subplot(1, 3, 3);
% imshow(edgeStrengthDifference, []);
% title('Edge Strength Difference');






%fprintf('RMSE=%f\n',resultImage);
%figure(6); imshow(resultImage,[]);title('edges');





% EXTRA CONTENT
% 
% ent_lp=entropy(uint8(imf))
% e=imt-imf;ent_e=entropy(uint8(e));
% figure(5); imshow(e,[]);title('error');
% mn_imf=mean(mean(double(imf)))
% % per_met1(L,:) = pereval_1(imt,abs(imf));
% % per_met2(L,:) = perevalwt(imf);
% 
% per_met2 = perevalwt(imf);
% per_met3 = pereval(imt,abs(imf));
% Q_ABF = Qp_ABF(im1,im2,imf)
% Q_abfe = Qabf_eval(im1,im2,imf)
% 
%  %Mean Square Error
%  MSE = MeanSquareError(imt, imf)
%  %Normalized Cross-Correlation 
%  NK = NormalizedCrossCorrelation(imt, imf)
%  %Average Difference 
%  AD = AverageDifference(imt, imf)
%  %Maximum Difference 
%  MD = MaximumDifference(imt, imf)
%  %Normalized Absolute Error
%  NAE = NormalizedAbsoluteError(imt, imf)
% 
% fusion_perform_fn(imf,imt);
%  %objective_fusion_perform_fn(imf,x)
% 
%  %ECC = fusionECC(im1,im2,imf)
% 
% %  [FSIM, FSIMc] = FeatureSIM(imt, imf)
% % 
% %  vifp_k = kun_vifp_mscale(imt,imf)
% % 
% %  NQM_k = kun_NQM_FR(imt,imf)
% 
%  
%  % e=imt-imf;
%  % RMSE = mean(e(:).^2);%root mean square error
%  % % for i=1:k
%  % %     figure(5+i);
%  % %     imshow(Idf{i},[]);%laplacian images
%  % % end
%  % fprintf('RMSE=%f\n',RMSE);
% 
% %  brisque_imt = brisque(imt)   % A smaller score indicates better perceptual quality. 
% %  brisque_imf = brisque(imf)
% 
% 
%  % model = brisqueModel
%  % score = brisque(imf,model)
% 
% 
% %  niqe_imt = niqe(imt)
% %  niqe_imf = niqe(imf)   %A smaller score indicates better perceptual quality.
% 
% 
% %  setDir = fullfile(toolboxdir('images'),'imdata');
% %  imds = imageDatastore(setDir,'FileExtensions',{'.jpg'});
% %  model = fitniqe(imds);
% %  fitniqe_imt = niqe(imt,model)
% %  fitniqe_imf = niqe(imf,model)
% % 
% %  % setDir = fullfile(toolboxdir('images'),'imdata');
% %  % imds = imageDatastore(setDir,'FileExtensions',{'.jpg'});
% %  % model = fitniqe(imds,'BlockSize',[48 96])
% %  % niqeimt = niqe(imt,model)
% %  % niqeimf= niqe(imf,model)
% % 
% % 
% % 
% %  model = niqeModel
% %  niqeModel_imt = niqe(imt,model)
% %  niqeModel_imf = niqe(imf,model)
% 
% nfmi = fmi(im1,im2,imf,'none',3)
% % % Or simply:
% % nfmi = fmi(ima,imb,imf);
% 
% % [q_S q_S_map] = piella_metric_q_s(im1,im2,imf);
% % fprintf('Piella metric Q_S: \n %f\n',q_S);
% 
% % [q_W q_W_map] = piella_metric_q_w(im1,im2,imf);
% % fprintf('Piella metric Q_W (weighted fusion quality index): \n %f\n',q_W);
% 
% 
% %%%%%%%%%%%% not required  %%%%%
% 
% 
% % 
% % q_S = piella_metric_q_s(im1,im2,imf)
% % 
% % q_W = piella_metric_q_w(im1,im2,imf)
% % 
% % q_E1 = piella_metric_q_e1(im1,im2,imf)
% % 
% % q_E2 = piella_metric_q_e2(im1,im2,imf)
% % 
% % q_C = cvejic_metric_q_c(im1,im2,imf)
% % 
% % q_Y = yang_metric_q_y(im1,im2,imf)
% % 
% % cq_M = pistonesi_metric_cq_m(im1,im2,imf)
% % 
% % 
% % % [q_E1 q_E1_map]= piella_metric_q_e1(im1,im2,imf);
% % % fprintf('Piella metric Q_E1 (edge-dependent fusion quality index (version 1)): \n %f\n',q_E1);
% % % 
% % % [q_E2 q_E2_map]= piella_metric_q_e2(im1,im2,imf);
% % % fprintf('Piella metric Q_E2 (edge-dependent fusion quality index (version 2)): \n %f\n',q_E2);
% % % 
% % % [q_C q_C_map] = cvejic_metric_q_c(im1,im2,imf);
% % % fprintf('Cvejic metric Q_C: \n %f\n',q_C);
% % % 
% % % [q_Y q_Y_map]= yang_metric_q_y(im1,im2,imf);
% % % fprintf('Yang metric Q_Y: \n %f\n',q_Y);
% % % 
% % % [cq_M cq_M_map]= pistonesi_metric_cq_m(im1,im2,imf); 
% % % fprintf('Pistonesi metric CQ_M: \n %f\n',cq_M);
% % % 
% % % 
% % % %res = fusionAssess(im1,im2,imf)
% % % 
% % % 
% % % 
% % 
% % % normalized mutual informtion $Q_{MI}$
% %     QMI=metricMI(im1,im2,imf,1)
% %     
% %     % Tsallis entropy $Q_{TE}$
% %     QTE=metricMI(im1,im2,imf,3)
% % 
% %     % Wang - NCIE $Q_{NCIE}$
% %     QNCIE=metricWang(im1,im2,imf)
% %     
% %     % Xydeas $Q_G$
% %     QG=metricXydeas(im1,im2,imf)
% %     
% %     % PWW $Q_M$
% %     QM=metricPWW(im1,im2,imf)
% %     
% %     % Yufeng Zheng (spatial frequency) $Q_{SF}$
% %     % QSF=metricZheng(im1,im2,imf)
% %     
% %     % Zhao (phase congrency) $Q_P$
% %     QP=metricZhao(im1,im2,imf)
% %     
% %     % Piella  (need to select only one) $Q_S$
% %     % Q(i,8)=index_fusion(im1,im2,fused);
% %     % QS=metricPeilla(im1,im2,imf,1)
% %     
% %     % Cvejie $Q_C$
% %     % QC=metricCvejic(im1,im2,imf,2)
% %     
% %     % Yang $Q_Y$
% %     % QY=metricYang(im1,im2,imf)
% %     
% %     % Chen-Varshney $Q_{CV}$
% %     QCV=metricChen(im1,im2,imf)
% %       
% %     % Chen-Blum $Q_{CB}$
% %     QCB=metricChenBlum(im1,im2,imf)
% %  
% % %     scd=SCD(im1,im2,imf)
% 
% %