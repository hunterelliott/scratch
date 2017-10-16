function [maxErrorPerLayer,maxErrorRatioPerLayer,maxErrorPerLayerParam,maxErrorRatioPerLayerParam] = cianVerifyGradients(layers,input,labels)
%CIANVERIFYGRADIENTS will use numerical differentiation to check the
%gradients of the input network.
%
% This may take several minutes to run depending on the computer and the
% size of the network. It will print progress as it checks each layer. You
% should expect the error magnitude in your gradients to be < 1e-4 for
% shallow networks.
%
% layers - a cell array containing the CIANLayer classes defining the
% network
%
% input - For an MLP this is a Num input dimensions x Num samples matrix of
%         input samples. 
%         For a CNN this is a Image height x image width x Num channels x
%         Num samples tensor of input image samples
%
% labels- A Num classes x Num samples logical matrix with the class labels
%         for each sample in a "one-hot" encoding - that is, 'true' in the
%         row corresponding to the class number for a sample and 'false'
%         elsewhere.
%
% Output:
% 
% Various measures of the error in the gradients at each layer. Note that
% numerical gradient calculation is not perfect so even if your gradients
% are correct the error will >0 but < 1e-4 for shallow networks.


epsilon = 1e-5;
nLayers = numel(layers);
activations = cianForward(layers,input,labels);
grads = cianBackward(layers);

gradsNumeric = cell(nLayers,1);
gradsNumericParams = cell(nLayers,1);

maxErrorPerLayer = nan(nLayers,1);
maxErrorRatioPerLayer = nan(nLayers,1);
maxErrorPerLayerParam =  nan(nLayers,1);
maxErrorRatioPerLayerParam =  nan(nLayers,1);

for iLayer = 1:nLayers        
    
    layerInput = activations{iLayer};
    gradsNumeric{iLayer} = nan(size(layerInput));
    
    for i = 1:numel(layerInput)
        dInputF = layerInput;
        dInputF(i) = layerInput(i) + epsilon/2;
        dInputB = layerInput;
        dInputB(i) = layerInput(i) - epsilon/2;
        
        act1 = cianForward(layers(iLayer:end),dInputF,labels);
        act2 = cianForward(layers(iLayer:end),dInputB,labels);
        deltaLoss = (act1{end} - act2{end})/epsilon;
                
        if any(deltaLoss ~= 0)
            gradsNumeric{iLayer}(i) = deltaLoss(deltaLoss ~= 0);%In case multiple samples input
        else
            gradsNumeric{iLayer}(i) = 0;
        end
        
    end
    maxErrorPerLayer(iLayer) = max(abs(gradsNumeric{iLayer}(:) - grads{iLayer}(:)));
    maxErrorRatioPerLayer(iLayer) = maxErrorPerLayer(iLayer) / mean(abs(gradsNumeric{iLayer}(:)));
    disp(['Maximum gradient error on ' class(layers{iLayer}) ', layer #'  num2str(iLayer) ' = ' num2str(maxErrorPerLayer(iLayer))])
    disp(['Maximum gradient error ratio on ' class(layers{iLayer}) ', layer #' num2str(iLayer) ' = ' num2str(maxErrorRatioPerLayer(iLayer))])
    
    %FINISH THIS!!
    if isa(layers{iLayer},'CIANParameterLayer')
        nParams = numel(layers{iLayer}.parameterFields);
        for j = 1:nParams
            gradPerSample = layers{iLayer}.(['grads' layers{iLayer}.parameterFields{j}]);
            nSamples = size(gradPerSample,ndims(gradPerSample));
            
            param = layers{iLayer}.(layers{iLayer}.parameterFields{j});
            
            gradsNumericParams{iLayer}{j} = nan([numel(param), nSamples]);
            
            
            for i = 1:numel(param)
                dParamF = param;
                dParamF(i) = param(i) + epsilon/2;
                layers{iLayer}.(layers{iLayer}.parameterFields{j}) = dParamF;
                act1 = cianForward(layers,input,labels);
                
                dParamB = param;
                dParamB(i) = param(i) - epsilon/2;
                layers{iLayer}.(layers{iLayer}.parameterFields{j}) = dParamB;
                act2 = cianForward(layers,input,labels);
                
                deltaLoss = (act1{end} - act2{end}) / epsilon;
                
                layers{iLayer}.(layers{iLayer}.parameterFields{j}) = param;
                
                gradsNumericParams{iLayer}{j}(i,:) = deltaLoss;                
                
            end
            
            gradsNumericParams{iLayer}{j} = reshape(gradsNumericParams{iLayer}{j},size(gradPerSample));
            
            maxErrorPerLayerParam(iLayer) = max(abs(gradsNumericParams{iLayer}{j}(:) - gradPerSample(:)));
            maxErrorRatioPerLayerParam(iLayer) = maxErrorPerLayerParam(iLayer) / mean(abs(gradsNumericParams{iLayer}{j}(:)));
            disp(['Maximum parameter gradient error on ' class(layers{iLayer}) ', layer #'  num2str(iLayer) ' = ' num2str(maxErrorPerLayerParam(iLayer))])        
            disp(['Maximum parameter gradient error ratio on ' class(layers{iLayer}) ', layer #'  num2str(iLayer) ' = ' num2str(maxErrorRatioPerLayerParam(iLayer))])
        end
        
        
    end
    
end


