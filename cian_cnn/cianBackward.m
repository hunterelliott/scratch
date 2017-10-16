function grads = cianBackward(layers)
%This function will perform a backwards pass on the network,
%back-propagating gradients from the loss to all the layers and parameters
%in the network.
%
% grads - the output of this network is a Num Layers x 1 cell array with
% the gradients for each layer.

nLayers = numel(layers);
grads = cell(nLayers,1);

grads{end} = layers{end}.backward();

for i = nLayers-1:-1:1    
    grads{i} = layers{i}.backward(grads{i+1});
    if isa(layers{i},'CIANParameterLayer')
        layers{i}.sideways(grads{i+1});
    end    
end
