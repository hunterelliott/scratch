function cianUpdate(layers,learningRate)

nLayers = numel(layers);

for i = 1:nLayers
    if isa(layers{i},'CIANParameterLayer')
        nParams = numel(layers{i}.parameterFields);
        for j = 1:nParams
            gradPerSample = layers{i}.(['grads' layers{i}.parameterFields{j}]);            
            paramGrad = mean(gradPerSample,ndims(gradPerSample));%Batch is always last dimension
            layers{i}.(layers{i}.parameterFields{j}) = layers{i}.(layers{i}.parameterFields{j}) - learningRate * paramGrad;
        end
    end
end
