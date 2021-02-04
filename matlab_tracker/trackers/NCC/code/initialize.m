function [ s ] = initialize( parameters, image, init_box )

%Shorthand for the parameters
s.p = parameters;

%%Coordinates are in the wrong order in initialization file
s.pos = [init_box(1), init_box(2)] + [init_box(3), init_box(4)] / 2;

%Size of the object that is tracked
s.size = init_box(3:4);

%Size of the template, including padding
s.template_sz = s.size + (s.size * s.p.padding);

%Generate the window weights for a 2D window
s.window = s.p.window_func(s.template_sz(1)) * s.p.window_func(s.template_sz(2))';


patch = get_region(image, s.pos, s.template_sz);
patch_gray = single(rgb2gray(patch));
patch_gray = patch_gray .* s.window;
patch_feature = (patch_gray / 255) - 0.5;

s.templatef = fft2(patch_feature);

end

