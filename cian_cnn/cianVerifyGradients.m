function [maxErrorPerLayer,maxErrorRatioPerLayer] = cianVerifyGradients(layers,input,labels)

epsilon = 1e-8;
nLayers = numel(layers);
activations = cianForward(layers,input,labels);
grads = cianBackward(layers);

gradsNumeric = cell(nLayers,1);
gradsNumericParams = cell(nLayers,1);

maxErrorPerLayer = nan(nLayers,1);
maxErrorRatioPerLayer = nan(nLayers,1);

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
    [maxErrorPerLayer(iLayer),iMax] = max(abs(gradsNumeric{iLayer}(:) - grads{iLayer}(:)));
    maxErrorRatioPerLayer(iLayer) = maxErrorPerLayer(iLayer) / abs(gradsNumeric{iLayer}(iMax));
    disp(['Maximum gradient error on layer ' num2str(iLayer) ' = ' num2str(maxErrorPerLayer(iLayer))])
    disp(['Maximum gradient error ratio on layer ' num2str(iLayer) ' = ' num2str(maxErrorRatioPerLayer(iLayer))])
    
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
            maxErrorPerLayerParam(iLayer) = max(abs(gradsNumericParams{iLayer}{j}(:) - gradPerSample(:)));
            disp(['Maximum parameter gradient error on layer ' num2str(iLayer) ' = ' num2str(maxErrorPerLayerParam(iLayer))])                        
        end
        
        
    end
    
end


