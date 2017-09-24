classdef CrossEntropyLayer
   properties
   end
   methods
       function obj = CrossEntropyLayer()
       end
   end
   methods (Static)
       function loss = forward(predictions,labels)
            %use logical indexing to avoid superflous log evaluations
            loss = -log(predictions(labels));
       end
       function dLoss_dPrediction = backward(predictions,labels)
           dLoss_dPrediction = zeros(size(predictions));
           dLoss_dPrediction(labels) = -1 ./ predictions(labels);
       end
   end        
end