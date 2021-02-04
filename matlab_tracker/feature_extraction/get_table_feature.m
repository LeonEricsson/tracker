function [ feature_map ] = get_table_feature( im, fparam)
%Get per-pixel features using a lookup table
%
% Usage
% feature_map = get_table_feature( im, fparam)
%
% Inputs:
% im        - input rgb image
% fparam    - struct containing feature parameters. Must have the
%             following fields:
%             'table' : Lookup table to be used for feature computation
%             'cell_size' : If greater than 1, the extracted features are
%             downsampled by computing cell wise averages, where a cell is 
%             a cell_size*cell_size patch in the image
%
% Outputs
% feature_map - Extracted feature map. Size of the feature map is equals
%               floor(size(im) / cell_size).

if isfield(fparam, 'cell_size')
    cell_size = fparam.cell_size;
else
    cell_size = 4;
end

% Extract features from table
feature_map = table_lookup(im, fparam.table);


if cell_size > 1
    feature_map = average_feature_region(feature_map, cell_size);
end

end

