function sequence_list = setup_sequences(env_parameters, sequence_list)
all_sequences = load_all_sequences(env_parameters);

if strcmp(sequence_list, 'otb_mini')
    sequence_list = all_sequences;
else
    seq_ids = zeros(numel(sequence_list),1);
    for i=1:1:numel(sequence_list)
        query_seq_id = find(cell2mat(cellfun(@(seq) strcmp(seq.name, sequence_list{i}),all_sequences, 'UniformOutput', false)));
        
        if isempty(query_seq_id)
            error(['Sequence not found: ', sequence_list{i}]);
        else
           seq_ids(i) = query_seq_id;
        end
    end
    sequence_list = all_sequences(seq_ids);
end
end

