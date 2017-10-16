classdef AveragePoolingLayer < CIANLayer    
    %This defines a pooling layer, wich performs averaging and downsampling
    %of the input images/feature maps.
   properties       
       poolSize
       inputSize
       padWidth
   end
   methods
       function obj = AveragePoolingLayer(poolSize)           
           obj.poolSize = poolSize; %Values will be averaged & pooled over poolSize x poolSize neighborhoods.
       end    
       function output = forward(obj,input)
           %The forward pass, performs average pooling on the input.
           %
           % Input - feature map heigh x feature map width x num feature
           % maps x num samples input.
           %
           
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
           
           %The output should be ~ 1/poolSize * (feature map height x
           %feature map width). Think carefully about how you handle images
           %that aren't a multiple of poolSize in height or width.
       end       
       function grads = backward(obj,gradNext)    
           %The backwards pass.
           
           % gradNext - an output height x output width x num feature maps
           % x num samples tensor of gradients
           
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
            
            %The output grads should be the same size as the layers inputs.
            %You will need to think carefully about how to match this up
            %with the forward pass for images that are not a multiple of
            %poolSize in height and width.
       end       
   end        
end
