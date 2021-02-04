function write_result( trk_name, param_name, seq_name, result, results_dir)

res_folder = [results_dir '/' trk_name];

if exist(res_folder,'dir') == 0
    mkdir(res_folder);
    mkdir([res_folder '/' param_name]);
end;

if exist([res_folder '/' param_name],'dir') == 0
    mkdir([res_folder '/' param_name]);
end;

fname = [res_folder '/' param_name '/' seq_name '.txt'];

dlmwrite(fname,result);

end

