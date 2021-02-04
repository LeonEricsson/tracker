function [ resized_patch ] = get_region( im, pos, sample_sz, output_sz )
%Get a region of pixels specified by pos and sample_sz from image im
%optional final argument for resampling to specified size
%
%Usage
%[ resized_patch ] = get_region( im, pos, sample_sz, [output_sz] )
% Inputs:
% im            -  Input image
% pos           -  Center of the region to be extracted [row, col]
% sample_sz     -  Size of the extracted region [num_rows, num_cols]
% output_sz     -  Size to which the extracted region is resized. Set to sample_sz if not provided 
%
% Outputs:
% resized_patch -  The extracted patch resized to output_sz

if nargin < 4
    output_sz = [];
end

%make sure the size is not to small
sample_sz = max(sample_sz, 2);

xs = floor(pos(2)) + (1:sample_sz(2)) - floor(sample_sz(2)/2);
ys = floor(pos(1)) + (1:sample_sz(1)) - floor(sample_sz(1)/2);

%check for out-of-bounds coordinates, and set them to the values at
%the borders
xs(xs < 1) = 1;
ys(ys < 1) = 1;
xs(xs > size(im,2)) = size(im,2);
ys(ys > size(im,1)) = size(im,1);

%extract image
im_patch = im(round(ys), round(xs), :);

if isempty(output_sz) || isequal(sample_sz(:), output_sz(:))
    resized_patch = im_patch;
else
    resized_patch = imresize(im_patch, output_sz, 'bilinear', 'Antialiasing',false);
end



end

