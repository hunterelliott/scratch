classdef SoftmaxLayer < handle
   properties
       activations
       J
   end
   methods
       function obj = SoftmaxLayer()           
       end    
       function output = forward(obj,logits)
            inputExp = exp(logits);
            output = bsxfun(@rdivide,inputExp,sum(inputExp));
            obj.activations = output;
       end       
       function J = jacobian(obj)
           [nActivations,nSamples] = size(obj.activations);
           % Should probably try to fully vectorize this...
           J = zeros(nActivations,nActivations,nSamples);           
           for i = 1:nSamples
               [Sj,Si] = meshgrid(obj.activations(:,i),obj.activations(:,i));
               J(:,:,i) = Si .* (eye(nActivations) - Sj);
           end
           obj.J = J;
       end
       function grads = backward(obj,gradNext)
            
           %Update the jacobian
            obj.jacobian();
            
            %Get gradients
            [nClasses,~,nSamples] = size(obj.J);
            grads = nan(1,nClasses,nSamples);
            for i = 1:nSamples
                grads(1,:,i) = gradNext(:,i)' * obj.J(:,:,i);
            end
           
       end
   end        
end