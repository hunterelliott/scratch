function cianUpdate(layers,learningRate)

nLayers = numel(layers);

for i = 1:nLayers
    if isa(layers{i},'CIANParameterLayer')
        nParams = numel(layers{i}.parameterFields);
        for j = 1:nParams            
            paramGrad = mean(layers{i}.(['grads' layers{i}.parameterFields{j}]),3);
            layers{i}.(layers{i}.parameterFields{j}) = layers{i}.(layers{i}.parameterFields{j}) - learningRate * paramGrad;
        end
    end
end
