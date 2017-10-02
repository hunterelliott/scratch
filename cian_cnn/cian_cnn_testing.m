%% --- parameters ---- %%

datasetName = 'MNIST'
networkType = 'CNN'

%% --- Define dataset --- %%



switch datasetName

    case 'Trivial'
        
        nClasses = 3;
        nSamples = 6;
        inputDims = 10;

        %Setup an easy classification problem

        labels = false(nClasses,nSamples);
        labels(1,1:2) = true;
        labels(2,3:4) = true;
        labels(3,5:6) = true;
        [~,labelInd] = max(labels,[],1);

        classMeans = linspace(0,nClasses*2,nClasses);
        input = randn(inputDims,nSamples);
        for j = 1:nSamples
            input(:,j) = input(:,j) + classMeans(labels(:,j));    
        end

        layerDims = [inputDims ./ 2 .^ (0:1), nClasses];
        
    case 'MNIST'
        
        mnistPath = fullfile(matlabroot,'toolbox','nnet','nndemos',...
                            'nndatasets','DigitDataset');
        mnistData = imageDatastore(mnistPath,...
                'IncludeSubfolders',true,'LabelSource','foldernames');

        labelInd = double(mnistData.Labels);
        nClasses = numel(unique(labelInd));
        nSamples = numel(labelInd);
        labels = false(nClasses,nSamples);
        labels(sub2ind(size(labels),labelInd',1:nSamples)) = true;
        input = mnistData.readall;
        
        switch networkType
            case 'MLP'
                input = cellfun(@(x)(x(:)),input,'Unif',0);                
                input = single([input{:}]);
                inputDims = size(input,1);
                
                layerDims = round([inputDims ./ 2 .^ (0:3), nClasses]);
                
            case 'CNN'
                input = single(cat(4,input{:}));
                inputDims = size(input,3);
                layerDims = [inputDims, round( 8 .* 2 .^ (0:2))];
                fcLayerDims = [layerDims(end) ./ 2 .^(0:1), nClasses];
                inShape = size(input);
        end
        
        input = input - 128;
        input = input / 128;
                
        
        
        
        
end
    
%% --- Define network --- %%



%Create simple MLP

biasInit = 0.1;
wVarInit = 0.01;
k = 3; %Kernel width

nLayers = numel(layerDims);
layers = {};
poolSize = 3;



switch networkType
    case 'MLP'            
        for iLayer = 1:nLayers-1            
            layers = [layers {InnerProductLayer(randn(layerDims(iLayer+1),layerDims(iLayer))*wVarInit,biasInit*ones(layerDims(iLayer+1),1))}];
            if iLayer < nLayers-1
                layers = [layers {ReLULayer()}];
            end
        end
        
    case 'CNN'
        
        nFCLayers = numel(fcLayerDims);
        for iLayer = 1:nLayers-1
            layers = [layers {ConvolutionalLayer(randn(k,k,layerDims(iLayer),layerDims(iLayer+1))*wVarInit,biasInit*ones(layerDims(iLayer+1),1))}];            
            layers = [layers {ReLULayer()}];
            layers = [layers {AveragePoolingLayer(poolSize)}];            
        end
        layers = [layers {FlattenLayer()}];
        
        %Run a forward pass as a hacky way to get FC starting dimension
        tmp = cianForward(layers,input(:,:,:,1:2),labels(:,1:2));
        fcLayerDims = [size(tmp{end},1), fcLayerDims]
        for iLayer = 1:nFCLayers
            layers = [layers {InnerProductLayer(randn(fcLayerDims(iLayer+1),fcLayerDims(iLayer))*wVarInit,biasInit*ones(fcLayerDims(iLayer+1),1))}];
            if iLayer < nFCLayers
                layers = [layers {ReLULayer()}];
            end
            
        end
        
end

layers = [layers {SoftmaxLayer()}];

layers = [layers {CrossEntropyLayer()}];

%% -- Test gradients --- %%

testInd = randsample(nSamples,2);

switch networkType 
    case'CNN'        
        
        testInput = input(:,:,:,testInd);
    case 'MLP'
        testInput = input(:,testInd);
end
    testLabels = labels(:,testInd);
    
cianVerifyGradients(layers,testInput,testLabels)



%% --- Train it ---- %%0

nIters = 1e3;
learningRate = 1e-2;
batchSize = 128;

lossPerIter = nan(nIters,1);
accPerIter = nan(nIters,1);

%Randomize sample order
sampleInds = randperm(nSamples);

for i = 1:nIters
    
    %Get the batch
    currIndInd = mod((i-1)*batchSize+1:i*batchSize,nSamples)+1;
    currInd = sampleInds(currIndInd);
    switch networkType
        case 'MLP'
            currInput = input(:,currInd);
        case 'CNN'
            currInput = input(:,:,:,currInd);
    end
    
    currLabels = labels(:,currInd);
    currLabelInd = labelInd(currInd);   
    
    %Make the forward and backward passes and the update
    activations = cianForward(layers,currInput,currLabels);
    preds = activations{end-1};
    losses = activations{end};
    gradInput = cianBackward(layers);
    cianUpdate(layers,learningRate)
    
    %Log performance
    lossPerIter(i) = mean(losses);    
    [~,predLabelInd] = max(preds,[],1);
    [~,predLabel] = max(preds,[],1);
    accPerIter(i) = double(nnz(predLabelInd(:) == currLabelInd(:))) / batchSize;
    
    if i == 1 || mod(i,10) == 0
        disp(['Iteration ' num2str(i) ', loss = ' num2str(lossPerIter(i)) ', accuracy = ' num2str(accPerIter(i))])
    end
end

%%

cf = figure;
subplot(2,1,1)
semilogy(lossPerIter)
ylabel('Loss')
subplot(2,1,2)
plot(accPerIter)
xlabel('Iteration')
ylabel('Accuracy')
%[preds,losses] = cianForward(layers,input,labels)