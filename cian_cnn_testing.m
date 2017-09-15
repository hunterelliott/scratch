
nConvLayers = 3;
nFCLayers = 3;
imSize = [28,28,1];
nClasses = 10;

inpuLayer = {imageInputLayer(imSize)};

convLayers = arrayfun(@(dim)(convolution2dLayer(3,dim)),8 .* 2 .^(0:nConvLayers),'Unif',false);
nonlinLayers = arrayfun(@(dim)(reluLayer()),0:nConvLayers,'Unif',false);
poolLayers = arrayfun(@(dim)(maxPooling2dLayer(2)),0:nConvLayers,'Unif',false);

convBlocks = vertcat(convLayers,nonlinLayers,poolLayers);


fcLayers = arrayfun(@(dim)(fullyConnectedLayer(dim)),[1024 ./ 2 .^(0:nFCLayers-1), nClasses],'Unif',false);
nonlinLayers = [arrayfun(@(dim)(reluLayer()),0:(nFCLayers-1),'Unif',false), {softmaxLayer()}];

fcBlocks = vertcat(fcLayers,nonlinLayers);

layers = [inpuLayer, convBlocks(:)' fcBlocks(:)' {classificationLayer()}]'

opts = trainingOptions('sgdm','ExecutionEnvironment','multi-gpu','MiniBatchSize',256);

mnistPath = fullfile(matlabroot,'toolbox','nnet','nndemos',...
    'nndatasets','DigitDataset');
mnistData = imageDatastore(mnistPath,...
        'IncludeSubfolders',true,'LabelSource','foldernames');

network = trainNetwork(mnistData,[layers{:}],opts);