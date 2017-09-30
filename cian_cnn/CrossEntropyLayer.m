classdef CrossEntropyLayer < handle
   properties
       activations
       labels
   end
   methods
       function obj = CrossEntropyLayer()           
       end
   end
   methods
       function loss = forward(obj,predictions,labels)
            %use logical indexing to avoid superflous log evaluations
            loss = -log(predictions(labels));
            obj.activations = predictions;
            obj.labels = labels;
       end
       function grads = backward(obj)
           grads = zeros(size(obj.activations));
           grads(obj.labels) = -1 ./ obj.activations(obj.labels);
       end
   end        
end