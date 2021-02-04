function feature_map = extract_features(image, pos, sample_sz, output_sz, features)
%Extract specified features from an image patch. The function extracts
%patches of size sample_sz centered at pos from the input image. The
%patches are then resized output_sz and the specified features are extracted 
%from the resized patch. The output has size output_sz divided by the cell size.
%
%Usage
%feature_map = extract_features(image, pos, sample_sz, output_sz, features)
%
% Inputs:
% image         -  Input image
% pos           -  Center of the patch to be extracted [row, col]
% sample_sz     -  Sizes of the extracted patches ([h1, w1; h2, w2; h3,w3; ...])
%                  Different sized patches can be extracted at the same time
% output_sz     -  Size to which extracted patches are resized before feature computation
% features      -  Cell array, where each element corresponds to a particular feature 
%                  to be extracted. Each element is a struct having the fields 'getFeature'and 'fparams'
%                  'getFeature' is a handle to the function which extracts
%                  the particular feature, while 'fparams' is a struct
%                  containing the parameter settings for that feature
%
%                  The function supports following feature types  
%                      1) Grayscale features
%                          getFeature = @get_grayscale_feature
%                          fparams    = struct('cell_size', CELL_SIZE)
%                      2) Look-up table features:
%                          getFeature = @get_table_feature
%                          fparams    = struct('tablename', TABLE_NAME, 'cell_size', CELL_SIZE)
%                      3) FHOG
%                          getFeature = @get_fhog
%                          fparams    = struct('cell_size', CELL_SIZE)                   
%
% Outputs:
%feature_map    - Cell array, where each element corresponds to a feature
%                 map. The size of each feature map depends on the type of 
%                 extracted feature and the input parameters  

num_features = length(features);
num_scales = size(sample_sz,1);

% Get image samples
image_patch = zeros(output_sz(1),output_sz(2), 3,num_scales,'uint8');

for i=1:1:num_scales
   image_patch(:,:,:,i) = get_region(image, pos, sample_sz(i,:), output_sz); 
end

feature_map = cell(1,1,num_features);

% Extract featues
num_feat_dim = 0;
for feat_ind = 1:num_features
    feat = features{feat_ind};
    
    % do feature computation
    feature_map{feat_ind} = feat.getFeature(image_patch, feat.fparams);
    num_feat_dim = num_feat_dim + size(feature_map{feat_ind},3);
end
              
end
