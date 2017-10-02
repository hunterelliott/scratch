function activations = cianForward(layers,input,labels)

nLayers = numel(layers);

activations = cell(nLayers+1,1);
activations{1} = input;

for i = 1:nLayers
    if isa(layers{i},'CrossEntropyLayer')        
        layerInputs = {activations{i}, labels};
    else
        layerInputs = activations(i);
    end
    activations{i+1} = layers{i}.forward(layerInputs{:});
end