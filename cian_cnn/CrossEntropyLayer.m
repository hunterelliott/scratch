classdef CrossEntropyLayer < CIANLayer
    %This defines a cross-entropy loss layer for classification tasks.
   properties
       inputs %We store the inputs and labels for use in the backward pass.
       labels
   end
   methods
       function obj = CrossEntropyLayer()           
       end
   end
   methods
       function loss = forward(obj,predictions,labels)
            %This forward method will calculate the loss for each sample
            %based on the predictions and labels (which will come from the
            %previous layer).
            
            %predictions - Num classes x Num samples matrix of predicted
            %probabilities (output of softmax layer)
            %labels - Num classes x Num samples logical matrix of labels,
            %with 'true' in the row corresponding to the class number for
            %that sample.
            
            loss = -log(predictions(labels));
             obj.inputs = predictions;
             obj.labels = labels;
             
             %The output of this method should be a Num samples x 1 vector
             %of the loss on each sample.
       end
       function grads = backward(obj)
           %The backwards pass for this layer. Since this is a loss layer
           %it doesn't need any input, but outputs the gradient of the loss
           %with respect to each samples input predictions.
           
           grads = zeros(size(obj.inputs));
           grads(obj.labels) = -1 ./ obj.inputs(obj.labels);

            %The output should be a Num classes x Num samples matrix of
            %gradients.
       end
   end        
end