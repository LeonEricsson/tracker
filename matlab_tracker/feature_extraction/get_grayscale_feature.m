function [ feature_map ] = get_grayscale_feature( im, fparam)
%%Extract grayscale features
%
% Usage
% feature_map = get_grayscale_feature( im, fparam)
% Inputs:
% im        - input rgb image
% fparam    - struct containing feature parameters. Must have the
%             following fields:
%             'cell_size' : If greater than 1, the extracted features are
%             downsampled by computing cell wise averages, where a cell is 
%             a cell_size*cell_size patch in the image
%
% Output:
% feature_map - Extracted feature map. Size of the feature map is equals
%               floor(size(im) / cell_size).

if isfield(fparam, 'cell_size')
    cell_size = fparam.cell_size;
else
    cell_size = 1;
end

% Extract features from table
feature_map = zeros(size(im,1),size(im,2),1,size(im,4));

for i=1:1:size(im,4)
    feature_map(:,:,:,i) =  single(rgb2gray(im(:,:,:,i)));
end

if cell_size > 1
    feature_map = average_feature_region(feature_map, cell_size);
end

end

