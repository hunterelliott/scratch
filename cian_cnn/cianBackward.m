function grads = cianBackward(layers)

nLayers = numel(layers);

grads = layers{end}.backward();
for i = nLayers-1:-1:1        
    gradsNext = layers{i}.backward(grads);
    if isa(layers{i},'CIANParameterLayer')
        layers{i}.sideways(grads);
    end
    grads = gradsNext;
end
