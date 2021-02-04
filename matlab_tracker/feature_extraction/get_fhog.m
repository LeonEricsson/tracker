function [ feature_image ] = get_fhog( im, fparam)
%Extract FHOG features using pdollar toolbox.
%
% Usage
% feature_image = get_fhog( im, fparam)
%
% Inputs:
% im          - input image.
% fparams     - struct containing feature parameters. Must have the
%               following fields:
%               'cell_size'  : The bin size used for fhog computation 
%
% Output:
% feature_image  - Extracted feature map. Size of the feature map is equals
%                  floor(size(im) / cell_size).

if ~isfield(fparam, 'nOrients')
    fparam.nOrients = 9;
end
if ~isfield(fparam, 'nDim')
    fparam.nDim = 31;
end
if isfield(fparam, 'cell_size')
    cell_size = fparam.cell_size;
else
    cell_size = 4;
end

[im_height, im_width, num_im_chan, num_images] = size(im);
feature_image = zeros(floor(im_height/cell_size), floor(im_width/cell_size), fparam.nDim, num_images, 'single');

for k = 1:num_images
    hog_image = fhog(single(im(:,:,:,k)), cell_size, fparam.nOrients);
    
    %the last dimension is all 0 so we can discard it
    feature_image(:,:,:,k) = hog_image(:,:,1:end-1);
end
end
