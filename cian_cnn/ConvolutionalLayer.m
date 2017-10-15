classdef ConvolutionalLayer < CIANParameterLayer
    %This defines a convolutional layer.
   properties
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
           
           obj.W = W; %W - a Kernel size x Kernel size x Num input channels x Num kernels tensor with the kernels for each output feature map
           obj.b = b; %b - a Num kernels x 1 vector of biases.
           obj.parameterFields = {'W','b'};
       end    
       function output = forward(obj,input)
           %The forward pass for this layer, which convolves all channels
           %of each of the input samples with each kernel to produce the
           %output feature maps/channels.
           %
           % Input - Image height x image width x Num channels x Num samples
           % tensor of input image samples / feature maps.
           %
           % Output - output height x output width x Num kernels x Num samples
           % tensor of output feature maps.
           %
           
           %Note that we need to use convn, but flip the kernel along the
           %3rd dimension. Why is that?
           
           %Also, let's avoid padding as it causes more trouble than it's
           %worth, even though our output feature maps will be smaller in
           %width and height than the inputs.
           
            nSamples = size(input,4);
            nKernels = size(obj.W,4);
            outputWH = [size(input,1),size(input,2)];
            padW = (size(obj.W,1)-1)/2;            
            output = zeros([outputWH-padW*2, nKernels, nSamples]);            
            paddedInput = input;
            
            for i = 1:nSamples
                for k = 1:nKernels
                    %We use n-d convolution as a shortcut, but need to
                    %pre-flip the kernel along the 3rd dimension so it is
                    %actually just a sum along the 2D kernel element
                    %convolutions
                    output(:,:,k,i) = convn(paddedInput(:,:,:,i),flip(obj.W(:,:,:,k),3),'valid') + obj.b(k);                    
                end
            end                        
            obj.inputs = input;
       end       
       function grads = backward(obj,gradNext)  
           %Backwards pass, converts gradNext - the gradients of the loss
           %with respect to this layers outputs, to grads - the gradients
           %of the loss with respect to this layers inputs.
           
           % gradNext - a output height x output width x Num kernels x Num samples
           % tensor of output feature maps.
            nSamples = size(gradNext,4);
            nKernels = size(obj.W,4);
            inDims = size(obj.inputs,3);
            grads = zeros(size(obj.inputs));

            for i = 1:nSamples
                for j = 1:inDims
                    for k = 1:nKernels                        
                        grads(:,:,j,i) = grads(:,:,j,i) + conv2(gradNext(:,:,k,i),rot90(obj.W(:,:,j,k),2),'full');
                    end
                end
            end
            % the output should be a Image height x image width x Num channels x Num samples
            % tensor of gradients.
           
       end
       function [gradsW,gradsb] = sideways(obj,gradNext)           
            nSamples = size(gradNext,4);            
            gradsW = nan([size(obj.W), nSamples]);
            gradsb = nan([size(obj.b), nSamples]); 
            inDims = size(obj.inputs,3);
            nKernels = size(obj.W,4);
                                    
            paddedInput = obj.inputs;
            
            for i = 1:nSamples
                for k = 1:nKernels
                    for j = 1:inDims                    
                        gradsW(:,:,j,k,i) = rot90(conv2(paddedInput(:,:,j,i),rot90(gradNext(:,:,k,i),2),'valid'),2);                        
                    end
                    gradsb(k,i) = sum(sum(gradNext(:,:,k,i)));
                end
            end
            obj.gradsW = gradsW;
            obj.gradsb = gradsb;
       end                      
   end        
end