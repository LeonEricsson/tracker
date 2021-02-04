function [ parameters ] = baseline(  )
%Default parameters for the NCC tracker

parameters = struct(...
'padding',1,...
'window_func',@hann);

end