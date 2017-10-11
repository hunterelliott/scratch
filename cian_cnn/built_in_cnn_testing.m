
nConvLayers = 3;
nFCLayers = 1;
%imSize = [28,28,1];
%nClasses = 10;
%imSize = [64,64,1];
%nClasses = 2;
poolSize = 2
kernelSize = 3;

nClasses = size(labels,1);
imSize = inShape(1:3);




%imageParentDir = '/media/hunter/Windows/shared/Data/DeadNet/Feb_12_EasySubset/TrainSet/preProc'
%imageParentDir = '/media/hunter/1E52113152110F61/shared/Data/DeadNet/Feb_12_EasySubset/TrainSet/preProc'
%imageParentDir = fullfile(matlabroot,'toolbox','nnet','nndemos','nndatasets','DigitDataset');


%inpuLayer = {imageInputLayer(imSize)};
inpuLayer = {imageInputLayer(imSize,'Normalization','none')};

convLayerDims = 8 .* 2 .^(0:nConvLayers);
convLayers = arrayfun(@(dim)(convolution2dLayer(kernelSize,dim)),convLayerDims,'Unif',false);
nonlinLayers = arrayfun(@(dim)(reluLayer()),0:nConvLayers,'Unif',false);
%poolLayers = arrayfun(@(dim)(maxPooling2dLayer(2)),0:nConvLayers,'Unif',false);
%poolLayers = arrayfun(@(dim)(maxPooling2dLayer(3)),0:nConvLayers,'Unif',false);
poolLayers = arrayfun(@(dim)(averagePooling2dLayer(poolSize,'Stride',[poolSize,poolSize])),0:nConvLayers,'Unif',false);

convBlocks = vertcat(convLayers,nonlinLayers,poolLayers);
convBlocks = convBlocks(1:end-1)


fcLayers = arrayfun(@(dim)(fullyConnectedLayer(dim)),[512 ./ 2 .^(0:nFCLayers-1), nClasses],'Unif',false);
%fcLayers = arrayfun(@(dim)(fullyConnectedLayer(dim)),[1024 ./ 2 .^(0:nFCLayers-1), nClasses],'Unif',false);
%fcLayers = arrayfun(@(dim)(fullyConnectedLayer(dim)),[32 ./ 2 .^(0:nFCLayers-1), nClasses],'Unif',false);
nonlinLayers = [arrayfun(@(dim)(reluLayer()),0:(nFCLayers-1),'Unif',false), {softmaxLayer()}];

fcBlocks = vertcat(fcLayers,nonlinLayers);

layers = [inpuLayer, convBlocks(:)' fcBlocks(:)' {classificationLayer()}]'

%opts = trainingOptions('sgdm','ExecutionEnvironment','multi-gpu','MiniBatchSize',16);
opts = trainingOptions('sgdm','MiniBatchSize',16,'L2Regularization',0.0,'Momentum',0.9,'InitialLearnRate',1e-2,'OutputFcn',@plotTrainingAccuracy,'MaxEpochs',5)


%Fix bias initialization

for iLayer = 1:numel(layers)
    
    if isa(layers{iLayer},'nnet.cnn.layer.Convolution2DLayer') 
        layers{iLayer}.Bias = ones(1,1,layers{iLayer}.NumFilters) * biasInit;
        %layers{iLayer}.Weights = randn(kernelSize,kernelSize,layers{iLayer}.NumFilters) * biasInit;
    end
    if isa(layers{iLayer},'nnet.cnn.layer.FullyConnectedLayer')
        layers{iLayer}.Bias = ones(layers{iLayer}.OutputSize,1) * biasInit;
    end
end


%%

%TEMP - using cian_cnn script to read data
%imageData = imageDatastore(imageParentDir,...
%        'IncludeSubfolders',true,'LabelSource','foldernames');
%network = trainNetwork(imageData,[layers{:}],opts);

network = trainNetwork(input,imageData.Labels,[layers{:}],opts);


