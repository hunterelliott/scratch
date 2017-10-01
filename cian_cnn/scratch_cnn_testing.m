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
dXedPred = lCE.backward()

%Verify numerically
epsilon = 1e-8;
%Just do the fast vectorized form
dXedPredNumeric = (lCE.forward(preds + epsilon/2,labels) - lCE.forward(preds - epsilon/2,labels)) / epsilon


%% ---- Softmax layer tests ---- %%

lSM = SoftmaxLayer();

% scores = [1   5.3 2.00;
%           1.5 -15  1.98;         
%           4   10  2.05]
scores = randn(3,3);      
%scores = lFC.forward(randn(4,3));

      
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
JsoftmaxNumeric
max(abs(JsoftmaxNumeric(:) - Jsoftmax(:)))

loss = lCE.forward(probs,labels);


gradsSM = lSM.backward(lCE.backward())
gradsSM



%Numerical verification of gradients
gradNumericSM = nan(1,nClasses,nSamples);
epsilon = 1e-10;
for j = 1:nClasses
    for s = 1:nSamples
        scoreDeltaF = scores;
        scoreDeltaB = scores;
        scoreDeltaF(j,s) = scoreDeltaF(j,s) + epsilon/2;
        scoreDeltaB(j,s) = scoreDeltaB(j,s) - epsilon/2;
        deltaLoss = (lCE.forward(lSM.forward(scoreDeltaF),labels) - lCE.forward(lSM.forward(scoreDeltaB),labels)) / epsilon;
        gradNumericSM(1,j,s) = deltaLoss(s);        
    end
end
gradNumericSM

    

max(abs(gradsSM(:)-gradNumericSM(:)))
%% ----- Fully-connected layer tests ------ %%

nClasses = 3;
nSamples = 3;
nNeurons = 4;
W = randn(nClasses,nNeurons);
b = randn(nClasses,1);

a = randn(nNeurons,nSamples);

lFC = InnerProductLayer(W,b);

scores = lFC.forward(a);

probs = lSM.forward(scores)
loss = lCE.forward(probs,labels)

gradsSM = lSM.backward(lCE.backward());

gradsFC = lFC.backward(gradsSM)

%Numerical verification of gradients
gradsNumericFC = nan(1,nNeurons,nSamples);
epsilon = 1e-11;
for k = 1:nNeurons
    for s = 1:nSamples
        aDeltaF = a;
        aDeltaB = a;
        aDeltaF(k,s) = aDeltaF(k,s) + epsilon/2;
        aDeltaB(k,s) = aDeltaB(k,s) - epsilon/2;
        deltaLoss = (lCE.forward(lSM.forward(lFC.forward(aDeltaF)),labels) - lCE.forward(lSM.forward(lFC.forward(aDeltaB)),labels)) / epsilon;
        gradsNumericFC(1,k,s) = deltaLoss(s);        
    end    
end
squeeze(gradsFC)
squeeze(gradsNumericFC)


[gradsW,gradsb] = lFC.sideways(gradsSM)

%Numerical verification of W updates
gradsNumericW = nan(nClasses,nNeurons,nSamples);
epsilon = 1e-11;
for j = 1:nClasses
    for k = 1:nNeurons
        for s = 1:nSamples
            WDeltaF = W;
            WDeltaB = W;
            WDeltaF(j,k) = WDeltaF(j,k) + epsilon/2;
            WDeltaB(j,k) = WDeltaB(j,k) - epsilon/2;
            lFC.W = WDeltaF;
            l1 = lCE.forward(lSM.forward(lFC.forward(a)),labels);
            lFC.W = WDeltaB;
            l2 = lCE.forward(lSM.forward(lFC.forward(a)),labels);
            deltaLoss = (l1 - l2) / epsilon;
            gradsNumericW(j,k,s) = deltaLoss(s);
        end    
    end
end
lFC.W = W;%Restore original value

gradsNumericW

max(abs(gradsW(:)-gradsNumericW(:)))
%%


%Numerical verification of bias updates
gradsNumericb = nan(nClasses,nSamples);
epsilon = 1e-11;
for j = 1:nClasses    
    for s = 1:nSamples
        bDeltaF = b;
        bDeltaB = b;
        bDeltaF(j) = bDeltaF(j) + epsilon/2;
        bDeltaB(j) = bDeltaB(j) - epsilon/2;
        lFC.b = bDeltaF;
        l1 = lCE.forward(lSM.forward(lFC.forward(a)),labels);
        lFC.b = bDeltaB;
        l2 = lCE.forward(lSM.forward(lFC.forward(a)),labels);
        deltaLoss = (l1 - l2) / epsilon;
        gradsNumericb(j,s) = deltaLoss(s);        
    end
end
gradsNumericb

max(abs(gradsb(:) - gradsNumericb(:)))


%% ---- ReLU layer ---- %%


lRU = ReLULayer();



%% ---- Conv layer --- %%


lC = ConvolutionalLayer(randn(3,3,3,5),ones(5,1));

im = imread('peppers.png');

out = lC.forward(im);


grads = lC.backward(randn(size(out)));

[gradW,gradb] = lC.sideways(randn(size(out)));

%% ---  pooling --- %%

lMP = AveragePoolingLayer(3);

out = lMP.forward(double(im)+randn(size(im)));








