function groundTruthImage = ProposedSfunc(inputImage)
    % Read the input image
    I = inputImage;

    [m, n, d] = size(I);
    if d == 3
        J = rgb2gray(I);
    else
        J = I;
    end
    
    pi = 3.14159;
   sig =3.14159 ; % default sigma value is 1
%     sig = (1.0 ./ (1.0 + exp(-3.14159)));
%     sig = (exp(0.5 * pi) - exp(-0.5 * pi)) ./ (exp(0.5 * pi) + exp(-0.5 * pi))


    % Ts_high = 1;
    % Ts_low = 0.4 * Ts_high;

%     Ts_high = 0.1;
%     Ts_low = 0.2 * Ts_high;


Ts_high = 0.1; %0.2
Ts_low = 0.1* Ts_high; %0.4

    Thresh = [Ts_low Ts_high];
    Cs = edge(J, 'canny', Thresh, sig);

    % Return the ground truth binary image
    groundTruthImage = Cs;
    
    % Display the result
%     figure, imshow(Cs);
end
