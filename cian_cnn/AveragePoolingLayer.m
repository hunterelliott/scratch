classdef AveragePoolingLayer < CIANLayer    
   properties       
       poolSize
       inputSize
       padWidth
   end
   methods
       function obj = AveragePoolingLayer(poolSize)
           obj.poolSize = poolSize;
           %assert(mod(poolSize,2)==1);%Don't want to deal with padding on even size pools
       end    
       function output = forward(obj,input)
           inDims = size(input,3);
           nSamples = size(input,4);
           pw = ceil((obj.poolSize-1)/2);
           output = zeros(size(input) - [2*pw 2*pw 0 nSamples]);           
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
            for i = 1:nSamples
                for j = 1:inDims
                    %Pad only on right and bottom to match sub-sampling
                    %above
                    grads(:,:,j,i) = padarray(kron(gradNext(:,:,j,i),ones(obj.poolSize)),2*[obj.padWidth,obj.padWidth],'post');
                end
            end            
       end       
   end        
end
