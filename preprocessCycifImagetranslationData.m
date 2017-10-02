%% --- Params ---- %%

%Training image params
imSize = [256,256];
nChan = 3;%Number of channels in output training data

bgThresh = 210; %All-channel threshold for simple background detection
maxBgFrac = .6;%Maximum fraction of background in patch for retention

addpath('/home/hunter/extern-repos/bfmatlab')



%% ---- Process H&E data ----- %%

wsiFile = '/media/hunter/1E52113152110F61/shared/Data/cycIF/H&E/training Batch3.vsi';
outputDir = '/media/hunter/1E52113152110F61/shared/Data/cycIF/Processed/HandE';

iPlane = 11;%Full res 20x stitched WSI plane

bfr = bfGetReader(wsiFile);

bfr.setSeries(iPlane)

wsiSize(1) = bfr.getSizeX();
wsiSize(2) = bfr.getSizeY();

stride = round(imSize * .75);

xSplits = 1:stride(1):wsiSize(1)-stride(1);
ySplits = 1:stride(2):wsiSize(2)-stride(1);

nPatchesWSI = (numel(xSplits)-1)*(numel(ySplits)-1);

patch = zeros([imSize, nChan],'uint8');

%%

for ix = 1:numel(xSplits)-1
    for iy = 1:numel(ySplits)-1
        for iChan = 1:nChan
            patch(:,:,iChan) = bfGetPlane(bfr,iChan,xSplits(ix),ySplits(iy),imSize(1),imSize(2));            
        end
        fracBg = nnz(all(patch>bgThresh,3))/prod(imSize);        
        if fracBg<maxBgFrac            
            outName = [outputDir filesep 'im_x' num2str(xSplits(ix),'%05.0f') '_y' num2str(ySplits(iy),'%05.0f') '.png'];
            imwrite(patch,outName)
        end
    end
    disp(ix)
end

%% ---- Process  CycIF  data ---- %%

ifImageDir = '/media/hunter/1E52113152110F61/shared/Data/cycIF/Fluorescence';
outputDirFluor = '/media/hunter/1E52113152110F61/shared/Data/cycIF/Processed/Fluorescence';
ifImages = dir([ifImageDir filesep '*.tif']);

%Get image size
bfr = bfGetReader([ifImageDir filesep ifImages(1).name]);

cycImSize(2) = bfr.getSizeX();
cycImSize(1) = bfr.getSizeY();
nChan = bfr.getImageCount;
chanUse = 1:nChan;
chanUse(mod(chanUse,4) == 1)= []; %Only use the first DAPI image
chanUse = [1 chanUse];
nChanUse = numel(chanUse);
hueVals = linspace(0,1,nChanUse);

border = 50;%Avoid registration artifacts in border areas
wSplits = 1:imSize(1):(cycImSize(1)-border);
hSplits = border:imSize(2):cycImSize(2);

for iImage = 1:numel(ifImages)
       
    bfr = bfGetReader([ifImageDir filesep ifImages(iImage).name]);
    im = zeros([cycImSize, 3, nChanCIF]);    
    for iChan = 1:nChanUse
        %Composite all channels into single RGB for lazy first test with
        %matching input and output dimensionality
        tmpIm = bfGetPlane(bfr,chanUse(iChan));        
        im(:,:,:,iChan)= hsv2rgb(cat(3,hueVals(iChan)*ones(cycImSize),mat2gray(tmpIm),ones(cycImSize)));
                
    end   
    %Min-project along channel axis
    im = min(im,[],4);
    
    
    %Crop and save tiles
    for iw = 1:numel(wSplits)-1
        for ih = 1:numel(hSplits)-1
            patch = im(hSplits(ih):hSplits(ih+1),wSplits(iw):wSplits(iw+1),:);                        
            outFile = [outputDirFluor filesep 'im' num2str(iImage,'%03.0f') '_w' num2str(iw) '_h' num2str(ih) '.png'];
            imwrite(uint8(patch*255),outFile)
        end
    end
    
    disp(iImage)
    
end
        

%% Train / test split

testFrac = .1;

dirsSplit = {outputDir,outputDirFluor};

for iDir = 2:numel(dirsSplit)
    
    ims = dir([dirsSplit{iDir} filesep '*.png']);
    
    nTest = ceil(numel(ims)*testFrac);
    iTest = randsample(numel(ims),nTest);
    testDir = [dirsSplit{iDir} '_test'];
    mkdir(testDir)
    for iIm = iTest'
        movefile([dirsSplit{iDir} filesep ims(iIm).name],testDir)
        disp(iIm)
    end
    
    
    
end
    