%% --- Params ---- %%

%Training image params
imSize = [256,256];
nChan = 3;%Number of channels in output training data

bgThresh = 210; %All-channel threshold for simple background detection

maxBgFrac = .75;%Maximum fraction of background in patch for retention
%maxBgFrac = .25;%Maximum fraction of background in patch for retention

addpath('/home/hunter/extern-repos/bfmatlab','/home/hunter/personal-repos/extern/ColorDeconvolutionMatlab')



%% ---- Process H&E data ----- %%

%wsiFile = '/media/hunter/1E52113152110F61/shared/Data/cycIF/H&E/training Batch3.vsi';
%outputDir = '/media/hunter/1E52113152110F61/shared/Data/cycIF/Processed/HandE';

wsiFile = '/media/hunter/1E52113152110F61/shared/Data/CK18/PhillipsTrain/CK18/6507.tiff';
outputDir = '/media/hunter/1E52113152110F61/shared/Data/CK18/PhillipsTrain/CK18_preProc';
isIHC = true;

%wsiFile = '/media/hunter/1E52113152110F61/shared/Data/CK18/PhillipsTrain/HandE/6345.tiff';
%outputDir = '/media/hunter/1E52113152110F61/shared/Data/CK18/PhillipsTrain/HandE_preProc';

%iPlane = 11;%Full res 20x stitched WSI plane
iPlane = 0;

bfr = bfGetReader(wsiFile);

bfr.setSeries(iPlane)



wsiSize(1) = bfr.getSizeX();
wsiSize(2) = bfr.getSizeY();

%stride = round(imSize * .75);
stride = imSize;


xMin = 4e3;
xMax = 87e3;
yMin = 12e3;
yMax = 85e3;
xSplits = xMin:stride(1):min(wsiSize(1)-stride(1),xMax);
ySplits = yMin:stride(2):min(wsiSize(2)-stride(1),yMax);

nPatchesWSI = (numel(xSplits)-1)*(numel(ySplits)-1);

%% --- Mask sampling positions --- %%

%CAN"T BECAUSE NOT PYRAMID

readTheWholeFuckingThing = false;

if readTheWholeFuckingThing
    tic
% im = zeros([wsiSize(2:-1:1) 3],'uint8');
% readChunkSize = 1e4;
% 
% readSplitsX = unique([1:readChunkSize:wsiSize(1),wsiSize(1)]);
% readSplitsY = unique([1:readChunkSize:wsiSize(2),wsiSize(2)]);
% tic
% for ixSplit = 1:numel(readSplitsX)-1
%     
%     for iySplit = 1:numel(readSplitsY)-1
%         
%         im(readSplitsY(iySplit:iySplit+1),readSplitsX(ixSplit:iXSplit+1),iChan) = bfGetPlane(bfr,iChan,readSplitsX(ixSplit),readSplitsY(iySplit),diff(readSplitsX(ixSplit:ixSplit+1)),diff(readSplitsY(iySplit:iySplit+1)));                
%         disp([ixSplit,iySplit])
%     end
%     
% end

%%


    im = imread(wsiFile);
    toc
end
%%

%outName = cell(numel(xSplits)-1,numel(ySplits)-1);
outName = cell(numel(xSplits)-1,1);
parfor ix = 1:numel(xSplits)-1
    parBfr = bfGetReader(wsiFile);    
    parBfr.setSeries(iPlane)

    patch = zeros([imSize, nChan],'uint8');
    outName{ix} = cell(numel(ySplits)-1);
    for iy = 1:numel(ySplits)-1
        for iChan = 1:nChan
            patch(:,:,iChan) = bfGetPlane(parBfr,iChan,xSplits(ix),ySplits(iy),imSize(1),imSize(2));            
        end
        fracBg = nnz(all(patch>bgThresh,3))/prod(imSize);        
        if fracBg<maxBgFrac            
            
            %Of foreground, we want to enrich IHC
            if isIHC
                H = rgb2hsv(patch);
                mask = H(:,:,2) > 0.28;

                fracIHCCurr = nnz(mask) / prod(numel(mask));    
                
                if fracIHCCurr > 0.2 || rand < 0.15
                    writeIt = true;
                else
                    writeIt = false;
                end
            else
                writeIt = true;
            end
            if writeIt
                outName{ix}{iy} = [outputDir filesep 'im_x' num2str(xSplits(ix),'%05.0f') '_y' num2str(ySplits(iy),'%05.0f') '.png'];
                        imwrite(patch,outName{ix}{iy})
            end
                    
            
            %outName = [outputDir filesep 'im_x' num2str(xSplits(ix),'%05.0f') '_y' num2str(ySplits(iy),'%05.0f') '.png'];
            %imwrite(patch,outName)
        end
        if mod(iy,50) == 0
            disp(iy)
        end
    end
    disp(ix)
    disp(outputDir)
end
% %Write TileConfiguration file for re-stitching the translated images
% wasWritten = reshape(~cellfun(@isempty,outName(:)),size(outName));
% [indX,indY] = find(wasWritten);
% writeTileConfigurationFile([outputDir filesep 'TileConfigurationFromMetadata.txt'],outName(wasWritten(:)),[xSplits(indX(:))',ySplits(indY(:))']);
% [~,imNames,~] = cellfun(@fileparts,outName(wasWritten(:)),'Unif',0);
% outTranslated = cellfun(@(x)(['fakeA_' x '.png']),imNames,'Unif',0);
% writeTileConfigurationFile([outputDir filesep 'TileConfigurationFromMetadataForTranslated.txt'],outTranslated,[xSplits(indX(:))',ySplits(indY(:))']);

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
        
%% -----  Creat CycIF tileconfig ---- %%

createTileConfigurationFromMetadata(ifImageDir,'ShowPlot',true,'NumDims',2)


%% ----- measure IHC distribution ---- %%



nSample = 1e3;

allIms = dir([outputDir  filesep '*.png']);
sampleIms = allIms(randsample(numel(allIms),nSample));

%%

fracIHC = nan(nSample,1);

allSampleIms = cell(nSample,1);

showPlots = false;

for iIm = 1:nSample%iHigh(10:10:50)'% 
        
    
    
    im = imread([outputDir filesep sampleIms(iIm).name]);
    
    allSampleIms{iIm} = im;
    imSize = size(im);
    %imHDAB = SeparateStains(im, RGBtoHDAB);
    %mask = imHDAB(:,:,2) < 0.8;
    H = rgb2hsv(im);
    mask = H(:,:,2) > 0.28;
    
    fracIHC(iIm) = nnz(mask) / prod(numel(mask));    
    
    if showPlots
        cf = figure;
        imshow(im);
        hold on
        spy(mask)
    end
        
    
    if mod(iIm,50) == 0
        disp(iIm);
    end
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
    