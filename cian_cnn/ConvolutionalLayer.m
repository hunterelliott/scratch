classdef ConvolutionalLayer < CIANParameterLayer
   properties
       activations
       inputs
       W
       b
       gradsW
       gradsb       
   end
   methods
       function obj = ConvolutionalLayer(W,b)
           assert(size(W,1)==size(W,2))
           assert(size(W,4)==size(b,1))
           assert(mod(size(W,1),2)==1) %Even-size kernels make padding annoying
           
           obj.W = W;
           obj.b = b;
           obj.parameterFields = {'W','b'};
       end    
       function output = forward(obj,input)
            nSamples = size(input,4);
            nKernels = size(obj.W,4);
            outputWH = [size(input,1),size(input,2)];
            output = zeros([outputWH, nKernels, nSamples]);
            
            padW = (size(obj.W,1)-1)/2;
            paddedInput = padarray(input,[padW,padW,0,0],'symmetric');
            
            for i = 1:nSamples
                for k = 1:nKernels                    
                    output(:,:,k,i) = convn(paddedInput(:,:,:,i),obj.W(:,:,:,k),'valid') + obj.b(k);                    
                end
            end            
            obj.activations = output;
            obj.inputs = input;
       end       
       function grads = backward(obj,gradNext)            
            nSamples = size(gradNext,4);
            nKernels = size(obj.W,4);
            inDims = size(obj.inputs,3);
            grads = zeros(size(obj.inputs));

            for i = 1:nSamples
                for j = 1:inDims
                    for k = 1:nKernels
                        grads(:,:,j) = grads(:,:,j) + conv2(gradNext(:,:,k),rot90(obj.W(:,:,j,k),2),'same');
                    end
                end
            end
           
       end
       function [gradsW,gradsb] = sideways(obj,gradNext)
            nSamples = size(gradNext,4);            
            gradsW = nan([size(obj.W), nSamples]);
            gradsb = nan([size(obj.b), nSamples]); 
            inDims = size(obj.inputs,3);
            nKernels = size(obj.W,4);
            
            padW = (size(obj.W,1)-1)/2;
            paddedInput = padarray(obj.inputs,[padW,padW,0,0],'symmetric');
            
            for i = 1:nSamples
                for k = 1:nKernels
                    for j = 1:inDims                    
                        gradsW(:,:,j,k) = rot90(conv2(paddedInput(:,:,j,i),rot90(gradNext(:,:,k,i),2),'valid'),2);                        
                    end
                    gradsb(k,i) = sum(sum(gradNext(:,:,k,i)));
                end
            end
            obj.gradsW = gradsW;
            obj.gradsb = gradsb;
       end                      
   end        
end