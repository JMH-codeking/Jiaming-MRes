function [impulse_t, h_l] = channel_gen( ...
    t, tau_l, a_l, Mt, theta, d, lambda...
)
%%% assume no beamforming in the design, so no beamforming vectors %%%

% parameters for creating the channel
    % t: time
    % a_l: attenuation of the channel
    % Mt, theta, d, lambda are steering vector paramters

    % f_doppler: doppler shift -> not required in this function

% output of the function
    % impulse_t: impulse response of the channel 
        % -> convolving with signal later
    % h_l: channel matrix describing attenuation and steering vectors

    steering_vec = steering_vec_gen(Mt, theta, d, lambda);
    h_l = a_l * steering_vec * steering_vec.'; 
    % this channel matrix of Mt x Mt is the channel matrix for each time stamp

    impulse_t = impulse(t, tau_l);

