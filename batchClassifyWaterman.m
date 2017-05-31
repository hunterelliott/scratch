function results = batchClassifyFMN2(imageDir,modelDir,varargin)
%BATCHCLASSIFYGROWTHINHIBITION classifies all bacterial images in a directory as growing or antibiotic inhibited
%
% Syntax:
%
% batchClassifyFMN2
% batchClassifyFMN2(imageDir,modelDir)
% batchClassifyFMN2(imageDir,modelDir,'OptionName1',optionValue1,...)
%
%
% Inputs:
%
%   imageDir - the directory containing all the images to classify.
%
%   modelDir - the directory containing the model to use for
%   classification. Should contain a deploy.prototxt, a virtual_batch.mat
%   and a model.caffemodel file.
%
%   Documentation of additional parameter name/value pairs can be found
%   within the input section of the function
%
%
% Outputs:
%
%   results - a table containing the prediction information for each image.
%
%   A spreadsheet containing all the classification results will also be written
%   to a directory adjacent to the image directory along with other misc results
%   files.
%   
% Hunter Elliott
% 4/2017


%% ---- Input ---- %%

if nargin < 1 || isempty(imageDir)
    imageDir = uigetdir(pwd,'Select a directory containing images to classify:');
end

if nargin < 2 || isempty(modelDir)
    modelDir = uigetdir(pwd,'Select a directory containing the classification model:');
end

% ----- Optional parameter name/value pairs ----- %

ip = inputParser;
ip.addParameter('OutputDirectory','',@ischar);%The directory to store outputs in. Default will be to create a new directory named after the image and model directories
ip.addParameter('Verbose',true,@islogical);%If true, progress and other information will be displayed during processing
ip.addParameter('MakeFigures',true,@islogical);%If true, an overlay of the prediction results on each image will be saved. Slows processing.
ip.addParameter('ShowFigures',false,@islogical);%If true, overlay figures will be displayed during processing (if this is false and MakeFigures is true they will still be created and saved, just not shown during processing)
ip.addParameter('SaveProgress',250);%If greater than zero, progress will be saved every n images
ip.addParameter('ResumeFromAutosave',false,@islogical);%If true and the SaveProgress was enabled, the progress will be resumed from the autosave file saved within the output directory (must specify the same input/output directories as the first time you ran it)
ip.addParameter('GPUNumber',0);%Which GPU to use. Defaults to using what caffe calls 1, which is actually device 0 and also the display GPU. Set to -1 to use CPU

ip.parse(varargin{:});

p = ip.Results;

%% ---- Init ----- %%



% ---- Directories ---- %

%Use the directory as the name of the image batch
imbatchName = imageDir(max(strfind(imageDir(1:end-1)+1,filesep)):end);
if ~strcmp(imageDir(end),filesep)
    imageDir = [imageDir filesep];
end
%Use the directory as the name of the model
modelName = modelDir(max(strfind(modelDir(1:end-1),filesep))+1:end);
if ~strcmp(modelDir(end),filesep)
    modelDir = [modelDir filesep];
end

%Setup the output directory
if isempty(p.OutputDirectory)
    p.OutputDirectory = [imageDir(1:end-1) '_classified_by_model_' modelName];
end

if ~strcmp(p.OutputDirectory(end),filesep)
    p.OutputDirectory = [p.OutputDirectory filesep];
end

if p.Verbose;disp(['Storing output in ' p.OutputDirectory]);end

mkdir(p.OutputDirectory)

overlayDir = [p.OutputDirectory 'Overlays' filesep];
if ~exist(overlayDir,'dir')
    mkdir(overlayDir)
end

% ---- Image files ---- %


imFiles = imDir(imageDir);
nIms = numel(imFiles);

if p.Verbose;disp(['Found ' num2str(nIms) ' images to classify in directory ' imageDir]);end

assert(nIms>0,'No images found to classify! Check the input directory!')

% ---- GPU ---- %

if p.GPUNumber >= 0
    caffe.set_mode_gpu()
    caffe.set_device(p.GPUNumber)
    if p.Verbose;disp(['Using GPU ' num2str(p.GPUNumber)]);end
else
    caffe.set_mode_cpu()
    if p.Verbose;disp('Using CPU');end
end


% ---- Model ----- %

modelFile = [modelDir 'model.prototxt'];
assert(exist(modelFile,'file')>0,['Cannot find model definition file ' modelFile ' in specified model directory!'])
weightFile = [modelDir 'weights.caffemodel'];
assert(exist(weightFile,'file')>0,['Cannot find weight file ' weightFile ' in specified model directory!'])
vbFile = [modelDir 'virtual_batch.mat'];
assert(exist(vbFile,'file')>0,['Cannot find virtual batch file ' vbFile ' in specified model directory!'])


if p.Verbose;tic;disp(['Loading model and virtual batch from directory ' modelDir]);end

net = caffe.Net(modelFile,weightFile,'test');
inShape = net.blobs('data').shape;%Get model input size

vb = load(vbFile);
vbName = fieldnames(vb);
vb = vb.(vbName{1});

if p.Verbose;disp(['Done, took ' num2str(toc) ' seconds.']);end



%% ---- Batch classification ---- %%

%Loop through all the images and classify them with moving window


if p.Verbose;tic;disp(['Starting classification...']);end

perTilePred = cell(nIms,1);


if p.ResumeFromAutosave    
    
    load([p.OutputDirectory 'result_autosave.mat']);
    startIm = min(find(cellfun(@isempty,perTilePred)))-1;
    if p.Verbose;disp(['Resuming progress from autosave file, which had ' num2str(startIm) ' completed images.']);end
else
    startIm = 1;
end



for iIm = startIm:nIms
    
    tic;
    
    % --- Prepare input ---- %
    
    %Load the image and determine the stride
    im = imread([imageDir imFiles(iIm).name]);
    imSize = size(im);
    imSize = imSize(1:2);
        
    predDims = ceil(imSize ./ inShape(1:2));%Number of tiles wide and high for non-overlapping classification
    
    %Set the stride so each prediction covers the same amount of the image,
    %but with minimal overlap
    stride = floor((imSize-inShape(1:2)) ./ (predDims-1));
    
    
    % --- run the tiled classification ---- %
    
    pred = caffe_tiledVirtualBatchClassification(im,net,vb,'Stride',stride,'PreProcessingFunction',@waterman_preProc);
    %We report results in terms of FMN2 probability
    pred = pred(:,:,2);
    
    perTilePred{iIm} = pred;
    
    
    % ---- Assemble output ---- %
    
    
    results.ImageName{iIm,1} = imFiles(iIm).name;
    results.MeanInhibitionProbability(iIm,1) = mean(pred(:));
    results.MedianInhibitionProbability(iIm,1) = median(pred(:));
    results.STDInhibitionProbability(iIm,1) = std(pred(:));
    results.FractionTilesAbovePt5(iIm,1) = nnz(pred(:)>.5)/numel(pred(:));
    
    
    if p.MakeFigures
        if p.ShowFigures
            cf = figure;
        else
            cf = figure('Visible','off');
        end
        %Up-sample the prediction to match the image size and overlay it
        predUpSamp = imresize(pred,imSize);                
        imshow(im)
        hold on
        ovHan = imshow(cat(3,zeros([imSize(1:2), 1]),ones(imSize(1:2)),zeros([imSize(1:2), 1])));
        ovHan.AlphaData = predUpSamp*.5;
        saveas(cf,[overlayDir imFiles(iIm).name(1:end-4) '_overlay.jpg'])
    end
    
    if p.Verbose;disp(['Finished image ' num2str(iIm) ' of ' num2str(nIms) ', took ' num2str(toc) ' seconds.']);end        
    
    if p.SaveProgress > 0 && mod(iIm,p.SaveProgress) == 0
        save([p.OutputDirectory 'result_autosave.mat'],'results','perTilePred','p','imageDir','modelDir')
    end
    
end


%% ----- Output ------ %%

results = struct2table(results);

%Write summary results in spreadsheet format
writetable(results,[p.OutputDirectory 'per_image_results.csv'])

%Write full results from all tiles to matlab format
save([p.OutputDirectory 'full_results.mat'],'results','perTilePred','p','imageDir','modelDir')




