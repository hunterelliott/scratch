

%% --- Cross-entropy layer tests ---- %%

l = CrossEntropyLayer();

%Use second axis as batch - each column is a sample
labels = logical([0  0  1; 
                  0  1  0; 
                  1  0  0]);

%First two right last one wrong
preds = [.1 .0 .3;
         .2 .9 .3;
         .7 .1 .4];

%Run forward and backward passes     
xe = l.forward(preds,labels)     
dXedPred = l.backward(preds,labels)

%Verify numerically
epsilon = 1e-8
dXedPredNumeric = (l.forward(preds + epsilon/2,labels) - l.forward(preds - epsilon/2,labels)) / epsilon


%% ---- Softmax layer tests ---- %%