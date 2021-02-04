%% Setup the enviroment variables
env_parameters = setup_env();

%% Set the list of trackers to run
tracker_list = {
    struct('name','NCC', 'parameters','baseline'),...
    ...struct('name','Your_awesome_tracker', 'parameters','baseline'),...
}; 

%% Set the sequences to be run
sequence_list = {'Soccer'};             % Runs only selcted sequences
% sequence_list = 'otb_mini';           % Runs all the sequences in the dataset

%% Run evaluation

% Display tracking output
visualize = 1;

evaluate_trackers(tracker_list, sequence_list, visualize, env_parameters);

%% Generate results
results = generate_results(tracker_list, sequence_list, env_parameters);