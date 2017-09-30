classdef (Abstract) CIANParameterLayer < CIANLayer
   properties   
       parameterFields
   end
   methods (Abstract)             
       grads = sideways(obj,gradNext)
   end        
end