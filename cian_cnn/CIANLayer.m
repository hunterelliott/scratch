classdef (Abstract) CIANLayer < handle
   properties (Abstract)
       activations       
   end   
   methods (Abstract)
        output = forward(obj,input)
        grads = backward(obj)               
   end        
end