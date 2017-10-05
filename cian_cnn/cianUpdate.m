function updates = cianUpdate(layers,learningRate,momentum,updatePrev)

nLayers = numel(layers);
updates = cell(nLayers,1);
for i = 1:nLayers
    if isa(layers{i},'CIANParameterLayer')
        nParams = numel(layers{i}.parameterFields);
        updates{i} = cell(nParams,1);
        for j = 1:nParams
            gradPerSample = layers{i}.(['grads' layers{i}.parameterFields{j}]);            
            paramGrad = mean(gradPerSample,ndims(gradPerSample));%Batch is always last dimension
            updates{i}{j} = learningRate * -paramGrad;
            if ~isempty(updatePrev) && momentum > 0
                layers{i}.(layers{i}.parameterFields{j}) = layers{i}.(layers{i}.parameterFields{j}) + updates{i}{j} + momentum * updatePrev{i}{j};
            else
                layers{i}.(layers{i}.parameterFields{j}) = layers{i}.(layers{i}.parameterFields{j}) + updates{i}{j};
            end
        end
    end
end
