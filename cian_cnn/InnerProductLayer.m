classdef InnerProductLayer < handle
   properties
       activations
       W
       b
   end
   methods
       function obj = InnerProductLayer(W,b)
           obj.W = W;
           obj.b = b;
       end    
       function output = forward(obj,input)
            nSamples = size(input,2);
            output = zeros(size(obj.W,1),nSamples);
            for j = 1:nSamples
                output(:,j) = obj.W * input(:,j) + obj.b;
            end            
            obj.activations = output;
       end       
       function grads = backward(obj,gradNext)            
            nSamples = size(gradNext,3);
            grads = nan(1,size(obj.W,2),nSamples);
            for i = 1:nSamples
                grads(:,:,i) = gradNext(:,:,i) * obj.W;               
            end
           
       end
   end        
end