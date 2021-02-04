% This script provides an example of extracting features from a deep
% network using matconvnet. Matconvnet uses two different formats to store
% networks, namely simplenn and dagnn. Older network architectures such as
% alexnet, vgg which have simple architecture (i.e. a number of layers 
% stacked one after another) are stored in simplenn format. The recent 
% architectures such as googlenet, resnet cannot be stored as simplenn
% since they have skip connections etc. Thus dagnn format was introduced to
% store such networks. This script provides an example of using both simplenn
% as well as dagnn

% NOTE: Architectures such as alexnet, vgg have fully connected layers
% which fixes the input size to be 224x224. Further some layers in these
% networks do not add padding. As a result some of the border regions are 
% ignored. To avoid these issues, we have modified alexnet and vgg-m by 
% i) Adding appropriate padding in all layers and ii) Removing fully
% connected layers. These modified networks can be used as feature
% extractors on images of any sizes. The modified versions of the networks
% are stored at  /site/edu/bb/TSBB17/networks along with googlenet and
% resnet. More pre-trained networks can be obtained at 
% http://www.vlfeat.org/matconvnet/pretrained/

%% Settings
% Path to the network file
net_path = '/site/edu/bb/TSBB17/networks/imagenet-googlenet-dag.mat';

% Whether the network is simplenn or dagNN
is_simplenn = false;

% Which network activation to use as features. 
% For dagNN, specify the name of the output. In the 'layers' struct array, 
% the field 'outputs' contains the name of the output for each layer.
% For simplenn, specify the index of the layer which you want to use. For
% e.g. for using activations of conv1 in vgg-m, set output_layer = 1
output_layer = 'conv1';

% Example for using simplenn network
% net_path = '/site/edu/bb/TSBB17/networks/imagenet-vgg-m-2048-edited.mat';
% is_simplenn = true;
% output_layer = 1;

% Read image and convert the datatype to single
im = imread('../demo_sequence/Soccer/img/0001.jpg');
im = single(im);

%% Load the network
if is_simplenn
    net = load(net_path);
else
    net_struct = load(net_path);
    net = dagnn.DagNN.loadobj(net_struct);
    net.mode = 'test';
    net.conserveMemory = false;     % If conserveMemory is true, then the intermediate activations of the network will not be saved
end

%% Normalize with average image
% Since the normalization image saved with the network is 224x224, resize
% it to match the image size
normalization_image = imresize(net.meta.normalization.averageImage, [size(im,1), size(im,2)]);
im = bsxfun(@minus, im, normalization_image);

%% Run the network
if is_simplenn
    cnn_feat = vl_simplenn(net, im, [], [], 'Mode', 'test');
else
    net.eval({'data',im});
end

% Extract output layer
if is_simplenn
    feature_map = cnn_feat(output_layer+1).x;   % 1 is added since cnn_feat(1) corresponds to input image
else
    output_var_idx = net.getVarIndex(output_layer);
    feature_map = net.vars(output_var_idx).value;
end

%% Plot the activations from the output_layer
figure(1);
num_channels = size(feature_map,3);
grid_sz = round(sqrt(num_channels));

for i=1:num_channels
    subplot('Position',[(mod(i-1,grid_sz))/grid_sz floor((grid_sz*grid_sz-i)/grid_sz)/grid_sz 1/grid_sz 1/grid_sz])
    imagesc(feature_map(:,:,i)); axis image; axis off;
end