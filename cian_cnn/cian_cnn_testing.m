%% --- Define dataset --- %%

datasetName = 'MNIST'

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
        input = cellfun(@(x)(x(:)),input,'Unif',0);
        
        input = double([input{:}]);
        input = input - 128;
        input = input / 128;
        
        inputDims = size(input,1);
        
        layerDims = round([inputDims ./ 2 .^ (0:3), nClasses]);
        
        
end
    
%% --- Define network --- %%

%Create simple MLP

biasInit = 0.1;
wVarInit = 0.01;

nLayers = numel(layerDims);
layers = {};

for iLayer = 1:nLayers-1

    layers = [layers {InnerProductLayer(randn(layerDims(iLayer+1),layerDims(iLayer))*wVarInit,biasInit*ones(layerDims(iLayer+1),1))}];
    
    if iLayer < nLayers-1
        layers = [layers {ReLULayer()}];
    end
end

layers = [layers {SoftmaxLayer()}];

layers = [layers {CrossEntropyLayer()}];




%% --- Train it ---- %%0

nIters = 1e3;
learningRate = 1e-2;
batchSize = 64;

lossPerIter = nan(nIters,1);
accPerIter = nan(nIters,1);

%Randomize sample order
sampleInds = randperm(nSamples);

for i = 1:nIters
    
    %Get the batch
    currIndInd = mod((i-1)*batchSize+1:i*batchSize,nSamples)+1;
    currInd = sampleInds(currIndInd);
    currInput = input(:,currInd);
    currLabels = labels(:,currInd);
    currLabelInd = labelInd(currInd);   
    
    %Make the forward and backward passes and the update
    [preds,losses] = cianForward(layers,currInput,currLabels);
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

cf = figure;
subplot(2,1,1)
semilogy(lossPerIter)
ylabel('Loss')
subplot(2,1,2)
plot(accPerIter)
xlabel('Iteration')
ylabel('Accuracy')
%[preds,losses] = cianForward(layers,input,labels)