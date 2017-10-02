classdef (Abstract) CIANLayer < handle
   properties (Abstract)       
   end   
   methods (Abstract)
        output = forward(obj,input)
        grads = backward(obj)               
   end        
end