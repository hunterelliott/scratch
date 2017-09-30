classdef InnerProductLayer < CIANParameterLayer
   properties
       activations
       inputs
       W
       b
       gradsW
       gradsb       
   end
   methods
       function obj = InnerProductLayer(W,b)
           obj.W = W;
           obj.b = b;
           obj.parameterFields = {'W','b'};
       end    
       function output = forward(obj,input)
            nSamples = size(input,2);
            output = zeros(size(obj.W,1),nSamples);
            for j = 1:nSamples
                output(:,j) = obj.W * input(:,j) + obj.b;
            end            
            obj.activations = output;
            obj.inputs = input;
       end       
       function grads = backward(obj,gradNext)            
            nSamples = size(gradNext,3);
            grads = nan(1,size(obj.W,2),nSamples);
            for i = 1:nSamples
                grads(:,:,i) = gradNext(:,:,i) * obj.W;               
            end
           
       end
       function [gradsW,gradsb] = sideways(obj,gradNext)
           nSamples = size(gradNext,3);
           gradsW = nan([size(obj.W), nSamples]);
           gradsb = nan([size(obj.b), nSamples]);
           for i = 1:nSamples
                gradsW(:,:,i) = gradNext(:,:,i)' * obj.inputs(:,i)';
                gradsb(:,i) = gradNext(:,:,i);
           end
           obj.gradsW = gradsW;
           obj.gradsb = gradsb;
       end                      
   end        
end