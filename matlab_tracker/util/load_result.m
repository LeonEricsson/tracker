function [result] = load_result( trk_name, param_name, seq_name, results_dir)
% Load results from the disk

fname = [results_dir '/' trk_name '/' param_name '/' seq_name '.txt'];

if ~exist(fname, 'file')
    error(['Error loading result. Required file: ' fname '  does not exists']);
end

result = dlmread(fname);

end

