function grads = cianBackward(layers)

nLayers = numel(layers);

%grads = layers{end}.backward();
grads = cell(nLayers,1);
grads{end} = layers{end}.backward();

for i = nLayers-1:-1:1    
    grads{i} = layers{i}.backward(grads{i+1});
    if isa(layers{i},'CIANParameterLayer')
        layers{i}.sideways(grads{i+1});
    end    
end
