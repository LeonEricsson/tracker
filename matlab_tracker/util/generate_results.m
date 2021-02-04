function detailed_results = generate_results(tracker_list, sequence_list, env_parameters)
%Compute tracking scores for the set of given trackers on the given
%sequences
%
% Usage
% detailed_results = generate_results(tracker_list, sequence_list, env_parameters)
%
% Inputs
% tracker_list   - The list of trackers to be evaluated. Should be a cell
%                  array, with each element being a struct with the fields
%                  'name' (the name of the tracker) and 'parameters' (the
%                  parameter setting file for the tracker)
% sequence_list  - Either a cell array containing the name of the sequences
%                  to be run, or 'otb_mini' in which case all the sequences
%                  are used
%
%Outputs
% detailed_results - A table which contains average overlap for each
%                    tracker on each sequence
%
% The script also displays the success plots

sequence_list = setup_sequences(env_parameters, sequence_list);

% Load results
tracker_names = cellfun(@(trk) trk.name, tracker_list, 'UniformOutput',false);
parameter_names = cellfun(@(trk) trk.parameters, tracker_list, 'UniformOutput',false);
sequence_names = cellfun(@(seq) seq.name, sequence_list, 'UniformOutput',false);

results = cell(length(tracker_list),1);
for n = 1:length(tracker_list)
    trk_name = tracker_names{n};
    param = tracker_list{n}.parameters;
    results{n} = cell(length(sequence_list),1);
    for m = 1:length(sequence_list)
        seq_name = sequence_names{m};
        results{n}{m} = load_result(trk_name,param,seq_name,env_parameters.results_dir);
    end
end

% Load ground truth
gt = cell(length(sequence_list),1);

for n = 1:length(sequence_list)
    seq_name = sequence_list{n}.name;
    gt{n} = dlmread(['anno/' seq_name '.txt']);
end

iou_all = cell(length(tracker_list),1);
ave_coverage_all = zeros(length(tracker_list),length(sequence_list));

% calculate overlap
for n = 1:length(tracker_list)
    iou_all{n} = cell(length(sequence_list),1);
    
    for m = 1:length(sequence_list)
        x1_A = results{n}{m}(:,1);
        x2_A = results{n}{m}(:,1) + results{n}{m}(:,3);
        
        y1_A = results{n}{m}(:,2);
        y2_A = results{n}{m}(:,2) + results{n}{m}(:,4);
        
        x1_B = gt{m}(2:end,1);
        x2_B = gt{m}(2:end,1) + gt{m}(2:end,3);
        
        y1_B = gt{m}(2:end,2);
        y2_B = gt{m}(2:end,2) + gt{m}(2:end,4);
        
        int_w = max(0, min(x2_A, x2_B) - max(x1_A, x1_B)); 
        int_h = max(0, min(y2_A, y2_B) - max(y1_A, y1_B));
        
        intersection_AB =  int_w.* int_h;
        area_A = results{n}{m}(:,3).*results{n}{m}(:,4);
        area_B = gt{m}(2:end,3).*gt{m}(2:end,4);
        
        iou_all{n}{m} = intersection_AB ./ (area_A + area_B - intersection_AB);
        ave_coverage_all(n,m) = mean(iou_all{n}{m});
    end
end


% Calculate overlap plot
overlap_plot_step = 0.01;
overlap_plot_values_all = zeros(length(tracker_list), length(sequence_list),(1/overlap_plot_step) + 1);

plot_x = [0:overlap_plot_step:1]';

for n = 1:length(tracker_list)
    for m = 1:length(sequence_list)
        num_frames = numel(iou_all{n}{m});
        for i=1:1:size(overlap_plot_values_all,3)
            current_overlap_threshold = plot_x(i);
            overlap_plot_values_all(n,m,i) = sum(iou_all{n}{m} > current_overlap_threshold) / num_frames;
        end
    end
end

overlap_plot_values = reshape(mean(overlap_plot_values_all,2),length(tracker_list),[]);
auc = squeeze(mean(overlap_plot_values, 2))*100;
[auc,sort_ids] = sort(auc, 'descend');

overlap_plot_values = overlap_plot_values(sort_ids,:);
tracker_names = tracker_names(sort_ids);
parameter_names = parameter_names(sort_ids);
ave_coverage_all = ave_coverage_all(sort_ids,:);

tracker_param_cell = cellfun(@(x,y) [x, ' ',y], tracker_names, parameter_names, 'uniformOutput', false);

% Replace _ with space
tracker_param_cell = cellfun(@(x) strrep(x, '_', ' '), tracker_param_cell, 'uniformOutput', false);


plot_legend = cell(1,length(tracker_names));

for i=1:length(tracker_names)
    plot_legend{i} = sprintf('%s:  [%.1f]',tracker_param_cell{i},auc(i));
end

detailed_results = array2table([ave_coverage_all, auc], 'VariableNames', [sequence_names, 'AUC'], 'RowNames',tracker_param_cell);

figure;
plot(plot_x, overlap_plot_values);
title('Success Plot')
xlabel('Overlap Threshold')
ylabel('Overlap Precision');
xlim([0 1]);
ylim([0 1]);
legend(plot_legend);
