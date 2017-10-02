classdef FlattenLayer < CIANLayer
   properties       
       inShape
   end
   methods
       function obj = FlattenLayer()           
       end    
       function output = forward(obj,input)
            obj.inShape = size(input);
            output = nan([prod(obj.inShape(1:3)),obj.inShape(4)]);
            for j = 1:obj.inShape(4)
                output(:,j) = reshape(input(:,:,:,j),[],1);
            end                        
       end       
       function grads = backward(obj,gradNext)
            grads = nan(obj.inShape);
            for j = 1:obj.inShape(4)
                grads(:,:,:,j) = reshape(gradNext(:,:,j),obj.inShape(1:3));
            end            
       end       
   end        
end