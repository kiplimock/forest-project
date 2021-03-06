% Auto-generated by cameraCalibrator app on 01-Feb-2022
%-------------------------------------------------------


% Define images to process
imageFileNames = {'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_11_51_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_18_01_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_18_31_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_18_42_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_18_48_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_18_53_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_19_01_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_19_06_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_19_10_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_19_13_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_19_17_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_19_21_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_19_38_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_19_41_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_19_43_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_19_46_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_19_50_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_19_54_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_19_59_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_20_07_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_20_09_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_20_11_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_20_17_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_20_21_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_20_25_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_20_27_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_20_31_Pro.jpg',...
    'D:\personal\dkut-msc\research thesis\learn opencv\images\cam1\take3\WIN_20220126_18_20_37_Pro.jpg',...
    };
% Detect checkerboards in images
[imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(imageFileNames);
imageFileNames = imageFileNames(imagesUsed);

% Read the first image to obtain image size
originalImage = imread(imageFileNames{1});
[mrows, ncols, ~] = size(originalImage);

% Generate world coordinates of the corners of the squares
squareSize = 23;  % in units of 'millimeters'
worldPoints = generateCheckerboardPoints(boardSize, squareSize);

% Calibrate the camera
[cameraParams, imagesUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
    'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'millimeters', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
    'ImageSize', [mrows, ncols]);

% View reprojection errors
h1=figure; showReprojectionErrors(cameraParams);

% Visualize pattern locations
h2=figure; showExtrinsics(cameraParams, 'CameraCentric');

% Display parameter estimation errors
displayErrors(estimationErrors, cameraParams);

% For example, you can use the calibration data to remove effects of lens distortion.
undistortedImage = undistortImage(originalImage, cameraParams);

% See additional examples of how to use the calibration data.  At the prompt type:
% showdemo('MeasuringPlanarObjectsExample')
% showdemo('StructureFromMotionExample')

save('cam.txt', 'cameraParams')
