function [ ...
    x, y, y_n, y_n_no_doppler, s, h_t, h_t_with_doppler ...
] = signal_generation( ...
    t, rou, M, fc, d, Mt, f_doppler_l, theta_l, b_l, tau_l ...
)
    %% parameter generation

    Q = rou * M; % oversampling factor -> number of time samples
    y = [];
    y_n = y;
    y_n_no_doppler = y;
    x = [];
    s = [];
    h_t_with_doppler = y;

    lambda = 3e8 / fc; % wavelength for steering vectors
    [impulse_t, h_t] = channel_gen(t, tau_l, b_l, Mt, theta_l, d, lambda);
    % this h_t & impulse_t is the channel matrix that is supposed to be LOS and static
    % h_t is Mr x Mt -> impulse_t is a delta function


    for i = 1:length(t) % same channel -> same tau, b, theta_l
        % signal generation 
        [x_t, s_t] = mimo_ofdm_comm_signal_gen(Q, M, Mt); % transpose -> create data matrix of Mt * Q
        x_t = x_t.'; s_t = s_t.'; % reshape the generated signals
        x(i, :, :) = x_t;
        s(i, :, :) = s_t;
        h_t_with_doppler(i, :, :) = h_t * exp(1j * 2*pi * f_doppler_l * t(i));
        y_org = h_t * x_t * exp(1j * 2*pi * f_doppler_l * t(i));
        y(i, :, :) = y_org; % y without time delay
        % Mt antennas, Q QPSK symbols once -> final matrix should be Mt x Q x t 
        %  -> reshape into two dimensions -> Mt x Q*time; 

        % pass AWGN channel
        y_n(i, :, :) = y_org + 0.1 * complex(randn(Mt, Q), randn(Mt, Q));
        y_n_no_doppler(i, :, :) =  ...
            y_n(i, :, :) / exp(1j * 2*pi * f_doppler_l * t(i));
        % remove the effect of doppler shift -> 
    end
    % time delay is equal to zero padding the entire generated y from the
    % left -> not required to be analysed directly


end





