function im = waterman_preProc(im,cropCent,cropSz)
%function to pre-process images for DeadNet forward passes. 
% im = waterman_preProc(im)
% im = waterman_preProc(im,cropCenter)
%
% All params hard coded...


if nargin < 2 || isempty(cropSz)    
    cropSz = [220 220];
end

imSz = size(im);
if nargin < 2 || isempty(cropCent)
    cropCent = round(imSz /2);    
    w = cropSz ./ 2; %Yeah I know, assuming even size, but it will ALMOST always likely be true....
end

im = im(cropCent(1)-w(1)+1:cropCent(1)+w(1),cropCent(2)-w(2)+1:cropCent(2)+w(2),:);

im = double(permute(im,[2 1 3])); %Shouldn't matter but transpose anyways...

%im = im - mean(im(:)); %Set mean to zero
im = im - 128; %Set mean to zero
%im = im ./ std(im(:)); %Set STD to 1
im = im ./ 20; %Set STD to 1





