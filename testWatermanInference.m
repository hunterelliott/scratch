%% --- load the model --- %%


%From scratch survey data net
model = '/ssd/Dropbox/NIH_Testing/Classification_Tests/Deadnet_on_5_16_2017_Data/deploy.prototxt';
weights =  '/ssd/Dropbox/NIH_Testing/Classification_Tests/Deadnet_on_5_16_2017_Data/snapshots/2017_05_31_0_iter_220000.caffemodel';



net = caffe.Net(model,weights,'test')

inShape = net.blobs('data').shape;
imSz = inShape(1:2);
batchSz = inShape(end);

%% --- init -- %%

caffe.set_mode_gpu()
caffe.set_device(0)


virtualBatchFile = '/ssd/Dropbox/NIH_Testing/Classification_Tests/Deadnet_on_5_16_2017_Data/virtual_batch.mat'

makeNewVirtualBatch = true;
if makeNewVirtualBatch

    %Path to training data for retreiving virtual batch
    imListFile =  '/ssd/NIH_Classification/Test/test.txt'
    vbImsIn = readtable(imListFile,'ReadVariableNames',false,'Delimiter',' ');
    
    batchSupportSize = batchSz-1; %Leave room for the test image
    
    % select 10 healthy and 10 sick images
    inhibID = find(vbImsIn.Var2);
    growID = find(not(vbImsIn.Var2));

    inhibID = inhibID(randperm(numel(inhibID)));
    inhibID = inhibID(1:floor(batchSupportSize/2));
    growID = growID(randperm(numel(growID)));
    growID = growID(1:floor(batchSupportSize/2));

    batchFileList = vbImsIn.Var1([inhibID; growID],:);


    % init imBatch, with empty first slot
    imBatch = NaN([imSz,3,batchSz]);

    for iIm = 1:batchSupportSize

        im = imread(batchFileList{iIm});
        im = waterman_preProc(im);
        imBatch(:,:,:,iIm+1) = im;

    end   
    
    save(virtualBatchFile,'imBatch')
else
    vb = load(virtualBatchFile,'imBatch')
    imBatch = vb.imBatch;
end



%% --- load a full image --- %%


%imageDir = '/ssd/Dropbox/KP_AST' %New test data

%Training/val data
imageDir = '/ssd/NIH_Classification/Test/A375M FMN2 crispr guide1-clone1 RGB' 
%imageDir = '/ssd/NIH_Classification/Test/A375M WT 2D Images RGB';
%imageDir = '/ssd/KP_Classification/Train/CPM_Inhibited_All'

imNames = imDir(imageDir);

nIms = numel(imNames);

iIm = randsample(nIms,1);
%iIm = 1;
imNames(iIm).name

im = imread([imageDir filesep imNames(iIm).name]);

fullImSize = size(im);
%% --- whole-image inference -- %%

tic
pred = caffe_tiledVirtualBatchClassification(im,net,imBatch,'PreProcessingFunction',@waterman_preProc);
toc

%% --- visualize ---- %%

predInhibUS = imresize(pred(:,:,2),fullImSize(1:2));
mean(mean(pred(:,:,2)))
cf = figure;
imshow(im)
hold on
ovHan = imshow(cat(3,zeros([fullImSize(1:2), 1]),ones(fullImSize(1:2)),zeros([fullImSize(1:2), 1])));
ovHan.AlphaData = predInhibUS*.5;


%% ---- Test batch processing ---- %%

modelDir = '/home/he19/files/CellBiology/IDAC/Hunter/Kirby/Smith/Classification_Models/From_Diverse_2_2017_Data';
%imageDir = '/ssd/KP_Classification/Batch_Proc_testSet_from_Train';
imageDir = '/ssd/KP_Classification/Batch_Proc_testSet_from_AST';

batchClassifyGrowthInhibition(imageDir,modelDir,'GPUNumber',0)


