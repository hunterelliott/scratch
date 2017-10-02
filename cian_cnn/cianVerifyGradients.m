function maxErrorPerLayer = cianVerifyGradients(layers,input,labels)

epsilon = 1e-8;
nLayers = numel(layers);

activations = cianForward(layers,input,labels);
grads = cianBackward(layers);

gradsNumeric = cell(nLayers,1);

maxErrorPerLayer = nan(nLayers,1);

for iLayer = 1:nLayers        
    
    input = activations{iLayer};
    gradsNumeric{iLayer} = nan(size(input));
    
    for i = 1:numel(input)
        dInputF = input;
        dInputF(i) = input(i) + epsilon/2;
        dInputB = input;
        dInputB(i) = input(i) - epsilon/2;
        
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
    disp(['Maximum gradient error on layer ' num2str(iLayer) ' = ' num2str(maxErrorPerLayer(iLayer))])
    
end


