classdef AveragePoolingLayer < CIANLayer    
   properties       
       poolSize
       inputSize
       padWidth
   end
   methods
       function obj = AveragePoolingLayer(poolSize)
           obj.poolSize = poolSize;           
       end    
       function output = forward(obj,input)
           inDims = size(input,3);
           nSamples = size(input,4);
           mod2 = mod(obj.poolSize,2);
           pw = ceil((obj.poolSize-mod2)/2);
           output = zeros(size(input) - [(1+mod2)*pw (1+mod2)*pw 0 nSamples]);
           for i = 1:nSamples
               for j = 1:inDims
                    output(:,:,j,i) = conv2(input(:,:,j,i),ones(obj.poolSize),'valid');
               end
           end    
           output = output(1:obj.poolSize:end,1:obj.poolSize:end,:,:);%This will discard remainder neighborhoods from right and bottom...           
           obj.inputSize = size(input);      
           obj.padWidth = pw;
       end       
       function grads = backward(obj,gradNext)    
            inDims = size(gradNext,3);
            nSamples = size(gradNext,4);
            grads = nan(obj.inputSize);
            gradNextSize = size(gradNext);
            pw = obj.inputSize(1:2) - obj.poolSize*gradNextSize(1:2);
            for i = 1:nSamples
                for j = 1:inDims
                    %Pad only on right and bottom to match sub-sampling
                    %above
                    grads(:,:,j,i) = padarray(kron(gradNext(:,:,j,i),ones(obj.poolSize)),pw,'post');
                end
            end            
       end       
   end        
end
