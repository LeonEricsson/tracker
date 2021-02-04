function evaluate_trackers(tracker_list, sequence_list, visualize, env_parameters)
%Evaluate a set of given trackers over a set of given sequences.
%
% Usage
% evaluate_trackers(tracker_list, sequence_list, visualize, env_parameters)
%
% Inputs
% tracker_list   - The list of trackers to be evaluated. Should be a cell
%                  array, with each element being a struct with the fields
%                  'name' (the name of the tracker) and 'parameters' (the
%                  parameter setting file for the tracker)
% sequence_list  - Either a cell array containing the name of the sequences
%                  to be run, or 'otb_mini' in which case all the sequences
%                  are used
% visualize      - If true, the tracker output is displayed

%% Verify the trackers and add some additional functions if missing
tracker_list = setup_trackers(env_parameters, tracker_list);

%% Load the sequences
sequence_list = setup_sequences(env_parameters, sequence_list);

%% Run each tracker over each sequence
for trk_no = 1:length(tracker_list)
    
    init = tracker_list{trk_no}.init;
    track = tracker_list{trk_no}.track;
    param = tracker_list{trk_no}.parameters;
    tracker_list{trk_no}.result = {};
    trk_name = tracker_list{trk_no}.name;
    
    disp(['Running tracker ' trk_name ' with parameter setting ' param ' on sequences: ']);
    
    addpath([env_parameters.path '/trackers/' trk_name '/parameters/']);
    addpath([env_parameters.path '/trackers/' trk_name '/code/']);
    
    for sequence_no = 1:length(sequence_list)
        seq_name = sequence_list{sequence_no}.name;
        disp(['  ' seq_name]);
        
        % If result already exists, skip
        result_path = [env_parameters.results_dir '/' trk_name '/' param '/' seq_name '.txt'];
        if exist(result_path,'file')
            disp('    Result already exists, skipping....');
            continue;
        end
        
        sequence_data = sequence_list{sequence_no};
        
        %Run a tracker on a sequence
        result = run_tracker(init,track,param,sequence_data,visualize);
        
        %Save the results to some file
        write_result(trk_name, param, seq_name, result, env_parameters.results_dir);
    end
    
    rmpath([env_parameters.path '/trackers/' trk_name '/parameters/']);
    rmpath([env_parameters.path '/trackers/' trk_name '/code/']);
end