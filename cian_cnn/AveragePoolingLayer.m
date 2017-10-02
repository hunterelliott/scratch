classdef AveragePoolingLayer < CIANLayer    
   properties       
       poolSize
       input
   end
   methods
       function obj = AveragePoolingLayer(poolSize)
           obj.poolSize = poolSize;
           assert(mod(poolSize,2)==1);%Don't want to deal with padding on even size pools
       end    
       function output = forward(obj,input)
           inDims = size(input,3);
           nSamples = size(input,4);
           output = zeros(size(input));
           for i = 1:nSamples
               for j = 1:inDims
                    output(:,:,j,i) = conv2(input(:,:,j,i),ones(obj.poolSize),'same');
               end
           end  
           %Keep track of which values were used for backprop           
           border = (obj.poolSize-1)/2;
           output = output(border+1:obj.poolSize:end,border+1:obj.poolSize:end,:,:);           
           obj.input = input;                       
       end       
       function grads = backward(obj,gradNext)    
            inDims = size(obj.input,3);
            nSamples = size(obj.input,4);
            grads = nan(size(obj.input));
            for i = 1:nSamples
                for j = 1:inDims
                    grads(:,:,j,i) = imresize(gradNext(:,:,j,i),size(obj.input(:,:,j,i)),'nearest');
                end
            end            
       end       
   end        
end
