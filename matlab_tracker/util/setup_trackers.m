function [ tracker_list ] = setup_trackers(env_parameters, tracker_list )

for trk_no = 1:length(tracker_list)
    trk_name = tracker_list{trk_no}.name;
    addpath([env_parameters.path '/trackers/' trk_name '/code']);
    addpath([env_parameters.path '/trackers/' trk_name '/parameters']);
    
    tracker_list{trk_no}.init = str2func('initialize');
    tracker_list{trk_no}.track = str2func('trackFrame');
    
    rmpath([env_parameters.path '/trackers/' trk_name '/code']);
    rmpath([env_parameters.path '/trackers/' trk_name '/parameters']);
end;

end

