classdef ReLULayer < CIANLayer
   properties
       activations       
   end
   methods
       function obj = ReLULayer()           
       end    
       function output = forward(obj,input)
            output = max(input,0);
            obj.activations = output;
       end       
       function grads = backward(obj,gradNext)                        
            grads = gradNext;
            grads(obj.activations==0) = 0;
       end       
   end        
end