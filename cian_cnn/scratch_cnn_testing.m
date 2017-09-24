clear all

%% --- Cross-entropy layer tests ---- %%

lCE = CrossEntropyLayer();

%Use second axis as batch - each column is a sample
labels = logical([0  0  1; 
                  0  1  0; 
                  1  0  0]);
[nClasses,nSamples] = size(labels);

%First two right last one wrong
preds = [.1 .0 .3;
         .2 .9 .3;
         .7 .1 .4];

%Run forward and backward passes     
xe = lCE.forward(preds,labels)     
dXedPred = lCE.backward(preds,labels)

%Verify numerically
epsilon = 1e-8;
%Just do the fast vectorized form
dXedPredNumeric = (lCE.forward(preds + epsilon/2,labels) - lCE.forward(preds - epsilon/2,labels)) / epsilon


%% ---- Softmax layer tests ---- %%

lSM = SoftmaxLayer();

scores = [1   5.3 20.0;
          1.5 15  19.8;         
          4   10  20.5]
[nClasses,nSamples] = size(scores);

probs = lSM.forward(scores)

Jsoftmax = lSM.jacobian()
sum(probs)


%Numerical varification of Jacobian
JsoftmaxNumeric = nan(nClasses,nClasses,nSamples);
epsilon = 1e-8;
for i = 1:nClasses
    for j = 1:nClasses
        for s = 1:nSamples
            scoreDeltaF = scores;
            scoreDeltaB = scores;
            scoreDeltaF(j,s) = scoreDeltaF(j,s) + epsilon/2;
            scoreDeltaB(j,s) = scoreDeltaB(j,s) - epsilon/2;
            deltaProbs = (lSM.forward(scoreDeltaF) - lSM.forward(scoreDeltaB)) / epsilon;
            JsoftmaxNumeric(i,j,s) = deltaProbs(i,s);            
        end
    end
end
disp(JsoftmaxNumeric)
max(abs(JsoftmaxNumeric(:) - Jsoftmax(:)))

grads = squeeze(lSM.backwards(dXedPred))


%Numerical verification of gradients
gradNumeric = nan(nClasses,nSamples);
epsilon = 1e-10;
for j = 1:nClasses
    for s = 1:nSamples
        scoreDeltaF = scores;
        scoreDeltaB = scores;
        scoreDeltaF(j,s) = scoreDeltaF(j,s) + epsilon/2;
        scoreDeltaB(j,s) = scoreDeltaB(j,s) - epsilon/2;
        deltaLoss = (lCE.forward(lSM.forward(scoreDeltaF),labels) - lCE.forward(lSM.forward(scoreDeltaB),labels)) / epsilon;
        gradNumeric(j,s) = deltaLoss(s);        
    end
end
gradNumeric