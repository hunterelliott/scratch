function [predictions,losses] = cianForward(layers,input,labels)

nLayers = numel(layers);

activations = input;
for i = 1:(nLayers-1)
    activations = layers{i}.forward(activations);
end
predictions = activations;
losses = layers{end}.forward(predictions,labels);