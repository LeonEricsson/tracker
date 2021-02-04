function [tracker_state ] = trackFrame(frame, tracker_state)
%Use normalized cross correlation to estimate the position shift for a
%patch relative to the previous frame.

    %Short version of state name to save typing
    s = tracker_state;
    
    patch = get_region(frame, s.pos, s.template_sz, s.template_sz);
    patch_gray = single(rgb2gray(patch));
    patch_gray = patch_gray .* s.window;
    %fake cross correlation normalization to save computations.
    patch_feature = (patch_gray / 255) - 0.5;
    patchf = fft2(patch_feature);
    
    %Calculate the response
    responsef = conj(patchf) .* s.templatef;
    response = real(ifft2(responsef));
    
    %Store the response for visualization
    s.response = response;
    
    %Calculate the maximum of the correlation
    [row, col] = ind2sub(s.template_sz, find(response == max(response(:)),1));
    
    %Update the position of the box according to the maximum
    %Since the maximum for zero offset is at the top left corner,
    %and the coordinate system wraps around we need some trickery
    %to extract the correct translation.
    translation_rows = mod(row-1+floor((s.template_sz(1)-1)/2),floor(s.template_sz(1))) - floor((s.template_sz(1)-1)/2);
    translation_cols = mod(col-1+floor((s.template_sz(2)-1)/2),floor(s.template_sz(2))) - floor((s.template_sz(2)-1)/2);
    
    s.pos = s.pos + [translation_rows, translation_cols];
    
    %Rolling average update
    %new_template = ((single(rgb2gray(get_region(frame, s.pos, s.template_sz, s.template_sz))) .* s.window) / 255) - 0.5;
    %s.template = new_template*0.9 + 0.1*new_template;
    
    
    %re-assign the state variable so we can pass it back in
    tracker_state = s;
end

