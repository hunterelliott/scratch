%% --- Define layer array --- %%

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


%Create simple MLP

layerDim = 6;
layers = {InnerProductLayer(randn(layerDim,inputDims),ones(layerDim,1))};

layers = [layers {ReLULayer()}];

layers = [layers {InnerProductLayer(randn(nClasses,layerDim),ones(nClasses,1))}];

layers = [layers {SoftmaxLayer()}];

layers = [layers {CrossEntropyLayer()}];




%% --- Train it ---- %%0

nIters = 1e4;
learningRate = 1e-4;

lossPerIter = nan(nIters,1);
accPerIter = nan(nIters,1);

for i = 1:nIters
    
    [preds,losses] = cianForward(layers,input,labels);
    gradInput = cianBackward(layers);
    cianUpdate(layers,learningRate)
    
    lossPerIter(i) = mean(losses);
    
    [~,predLabelInd] = max(preds,[],1);
    [~,predLabel] = max(preds,[],1);
    accPerIter(i) = nnz(predLabelInd == labelInd) / nSamples;
    
    if mod(i,100) == 0
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
[preds,losses] = cianForward(layers,input,labels)