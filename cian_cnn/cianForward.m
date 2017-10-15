function activations = cianForward(layers,input,labels)
%This function performs a forward pass on the input network, producing
%activations, where the "activations" of the second-to last layer are the
%predictions and the "activations" of the last layer are the losses. 
%
% activations = cianForward(layers,input,labels)
%
% layers - a cell array containing the CIANLayer classes defining the
% network
%
% input - For an MLP this is a Num input dimensions x Num samples matrix of
%         input samples. 
%         For a CNN this is a Image height x image width x Num channels x
%         Num samples tensor of input image samples
%
% labels- A Num classes x Num samples logical matrix with the class labels
%         for each sample in a "one-hot" encoding - that is, 'true' in the
%         row corresponding to the class number for a sample and 'false'
%         elsewhere.
%
% Output:
%
% activations - a cell array of the activations. Note that we use the
% convention where activations{i} is the activations that are INPUT to
% layers{i} so activations{i} is actually the activations of layer{i-1}

nLayers = numel(layers);

activations = cell(nLayers+1,1);
activations{1} = input;

%Loop through the layers to perform the forward pass. Note that the
%CrossEntropyLayer is a special case.

for i = 1:nLayers
    if isa(layers{i},'CrossEntropyLayer')        
        layerInputs = {activations{i}, labels};
    else
        layerInputs = activations(i);
    end
    activations{i+1} = layers{i}.forward(layerInputs{:});
end