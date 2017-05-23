%% --- train / test split ---- 


rgbDirs = {'A375M FMN2 crispr guide1-clone1 RGB',...
    'A375M FMN2 crispr guide2-Clone1 RGB',...
    'A375M WT 2D Images RGB',...
    'WT A375M treated with 100nm Cyto D for 30mins RGB'};
nDir = numel(rgbDirs);

parentDir = '/ssd/NIH_Classification/';
testDir = '/ssd/NIH_Classification/Test';
trainDir = '/ssd/NIH_Classification/Train';

testFrac = .2;

for iDir = 1:nDir
    
    
    inIms = imDir([parentDir filesep rgbDirs{iDir}]);
    
    nIms = numel(inIms)
    
    iTest = randsample(nIms,ceil(nIms*testFrac));
    
    isTest = false(nIms,1);
    isTest(iTest) = true;
    
    for iIm = 1:nIms
        
        if isTest(iIm)
            currOutDir = testDir;
        else
            currOutDir = trainDir;
        end
        
        currOutFull = [currOutDir filesep rgbDirs{iDir}];
        mkdir(currOutFull)        
        disp(['moving ' [parentDir filesep rgbDirs{iDir} filesep inIms(iIm).name] ' to ' currOutFull])
                
        movefile([parentDir filesep rgbDirs{iDir} filesep inIms(iIm).name],[currOutDir filesep rgbDirs{iDir} filesep])
    end
end

%% --- crop / aug --- 

%Test
allTestIms = caffe_preProcessFolders(testDir,{'WT','crispr','Cyto'},'windowSize',[256 256],'stride',[128 128],'NormMean',128,'NormVar',20^2,'varThresh',100,'OutputFile',[testDir filesep 'test.txt']);


%% Train - augmented

allTrainIms = caffe_preProcessFolders(trainDir,{'WT','crispr','Cyto'},'windowSize',[256 256],'stride',[128 128],'NormMean',128,'NormVar',20^2,'varThresh',100,'numTPS',4,'numBlur',2,'numRotate',4,'augSpacing',96,'warpSize',6,'blurRange',[1.0 1.15],'OutputFile',[trainDir filesep 'train_aug.txt']);


    