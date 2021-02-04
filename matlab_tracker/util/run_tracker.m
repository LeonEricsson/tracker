function sequence_output = run_tracker( init_func, tracking_func, param_file, sequence_data, visualize)
%Run the tracker on the the specified sequence, using the specified init
%function, with parameters from the specified parameter file

sequence_output = zeros(sequence_data.len,4);

for frame_no = 1:sequence_data.len
    
    image = imread(sequence_data.frame_filenames{frame_no});
    
    if frame_no == 1
        parameters = eval(param_file);
        tracker_state = init_func(parameters, image, sequence_data.init_rect);
        
        if visualize
            rsz = tracker_state.size(2:-1:1);
            rpos = tracker_state.pos(2:-1:1) - rsz/2;
            rect_position = [rpos, rsz];
            im_handle = imshow(image, 'Border','tight', 'InitialMag', 100);
            hold on;
            rect_handle = rectangle('Position',rect_position, 'EdgeColor','red', 'LineWidth', 2);
            drawnow;
        end;
        
    else
        tracker_state = tracking_func(image, tracker_state);
        
        if visualize
            rsz = tracker_state.size(2:-1:1);
            rpos = tracker_state.pos(2:-1:1) - rsz/2;
            rect_position = [rpos, rsz];
            set(im_handle, 'CData', image)
            set(rect_handle, 'Position', rect_position)
            drawnow;
        end;
    end;
    sequence_output(frame_no,[2 1 4 3]) = [tracker_state.pos - 0.5*tracker_state.size, tracker_state.size];    
    %Save the position
end;
close all;