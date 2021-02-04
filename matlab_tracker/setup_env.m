function [ env_settings ] = setup_env
%% Add the path to sequences and results

pathstr = mfilename('fullpath');
[toolkit_path,~,~]= fileparts(pathstr);

% Path to the directory where the sequences are stored
% for lab computers, use /site/edu/bb/TSBB17/otb_mini
env_settings.sequence_dir = [toolkit_path,'/demo_sequence']; 


% Path to the directory where you want to save the results
env_settings.results_dir = [toolkit_path,'/results'];

%% Compiles external libraries and sets up the paths

if isempty(env_settings.sequence_dir) || isempty(env_settings.results_dir)
    error('Paths to sequences and results not set! Please set them in setup_env.m');
end

pathstr = mfilename('fullpath');
fidx = strfind(pathstr,'/');
path = pathstr(1:fidx(end));

env_settings.path = path;

%Remove annoying error message
warning('off','all');
addpath(genpath([env_settings.path '/external/']));
addpath(genpath([env_settings.path '/feature_extraction/']));
addpath(genpath([env_settings.path '/util/']));

installed_marker = [path '/external/installed'];

if ~exist(installed_marker,'file')
    disp('First run, compiling mexfiles!');
    
    %Piotrs toolbox
    toolboxCompile;
    %matconvnet
    vl_compilenn;
    
    % Move mex files to matlab folder
    matconvnet_path = [env_settings.path '/external/matconvnet/matlab/'];
    movefile([matconvnet_path 'mex/vl_*.mex*'], matconvnet_path);
    
    %Place an empty file there to note that compilation has already been
    %done
    fclose(fopen(installed_marker,'w'));
end

warning('on','all');
end
