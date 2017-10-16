classdef InnerProductLayer < CIANParameterLayer
    %This class defines an inner product layer, also known as a
    %fully-connected layer. This layer has trainable parameters so it is a
    %sub-class of CIANParameterLayer
   properties
       %The class has properties we will use to store the input, the
       %parameters, and the parameter gradients.
       inputs
       W
       b
       gradsW
       gradsb       
   end
   methods
       function obj = InnerProductLayer(W,b)           
           obj.W = W; %W - Num input neurons x N output neurons weight matrix
           obj.b = b; %b - Num output neurons x 1 column vector of biases
           obj.parameterFields = {'W','b'}; %List of trainable parameter properties.
       end    
       function output = forward(obj,input)
           %This defines the forward pass for this layer, used when making
           %predictions.
           % Input: Num input neurons/dimensions x Num samples input matrix.
           
            %Loop through the samples, apply the linear transformation and
            %biases to complete the forward pass.
            nSamples = size(input,2);
            output = zeros(size(obj.W,1),nSamples);
            for j = 1:nSamples
                output(:,j) = obj.W * input(:,j) + obj.b;
            end
            %This layer needs the inputs for its backwards pass so we store
            %those here.
            obj.inputs = input;
            
            %The output of this layer is a Num output neurons x Num samples
            %matrix, or the "activations" of this layer.
       end       
       function grads = backward(obj,gradNext)            
           %This defines the backwards pass for this layer, used when
           %calculating gradients to be used during training.
            
           %gradNext - 1 x Num output neurons x Num samples matrix of gradients from the next layer.
            
            %Loop through the samples and propagate the gradient from the
            %outputs of this layer to it's inputs.
            nSamples = size(gradNext,3);
            grads = nan(1,size(obj.W,2),nSamples);
            for i = 1:nSamples
                grads(:,:,i) = gradNext(:,:,i) * obj.W;               
            end
           
            %The output of this layer is a 1 x Num input neurons x Num
            %samples matrix. 
       end
       function [gradsW,gradsb] = sideways(obj,gradNext)
           %This method will calculate the gradients for the parameters for
           %this layer, so they can be updated during training.
           
           %We calcualte separate gradients for each sample, but they will
           %be averaged during training.
           nSamples = size(gradNext,3);
           gradsW = nan([size(obj.W), nSamples]);
           gradsb = nan([size(obj.b), nSamples]);
           for i = 1:nSamples
                gradsW(:,:,i) = gradNext(:,:,i)' * obj.inputs(:,i)';
                gradsb(:,i) = gradNext(:,:,i);
           end
           %We store the gradients an a field so they can be used by the
           %update function. These gradients are the same size as the
           %corresponding parameters.
           
           obj.gradsW = gradsW;
           obj.gradsb = gradsb;
       end                      
   end        
end