classdef CrossEntropyLayer < CIANLayer
   properties
       inputs
       labels
   end
   methods
       function obj = CrossEntropyLayer()           
       end
   end
   methods
       function loss = forward(obj,predictions,labels)
%             %use logical indexing to avoid superflous log evaluations
            loss = -log(predictions(labels));
%             loss = -sum(log(predictions) .* labels,1);
             obj.inputs = predictions;
             obj.labels = labels;
       end
       function grads = backward(obj)
           grads = zeros(size(obj.inputs));
           grads(obj.labels) = -1 ./ obj.inputs(obj.labels);
%           grads = (-1 ./ obj.inputs) .* obj.labels;
       end
   end        
end