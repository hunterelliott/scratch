classdef ReLULayer < CIANLayer
    %A layer for the rectified linear unit nonlinearity.
   properties
       %We will store the activations in this property after the forward
       %pass so we can use them in the backward pass.
       activations       
   end
   methods
       function obj = ReLULayer()           
       end    
       function output = forward(obj,input)
           %The forward pass method. This takes in input and applies the
           %nonlinearity element-wise.
            output = max(input,0);
            obj.activations = output;
       end       
       function grads = backward(obj,gradNext)                        
           %The backward pass will take in gradients from the next layer
           %and output gradients of this layer. That is, it converts
           %gradients of the loss with respect to this layers output into
           %gradients of the loss with respect to this layers input.
            grads = gradNext;
            grads(obj.activations==0) = 0;
       end       
   end        
end