clc; clear;
%% signal initialisation
for i = 1:4000
    theta = pi / 3;
    phi = theta; % steering angles
    Nr = 4; % receive antennas
    Nt = 1; % transmit antennas
    Q = 1; % one user
    M = 16; % number of subcarrier
    K = 200; % 100 OFDM symbols
    fc = 3e11; % assume central frequency is 3 x 10^8 MHz
    lambda = 3e8 / fc;
    d = lambda / 2;
    scs = 20e3;        % Subcarrier spacing in Hz
    Fs = scs * M/2; % Sampling rate (1.28e6 Hz)
    Ts = 1 / Fs;       % Sample duration in seconds  
    
    L = 1; % assume line of sight -> only one path, no multipath
    f_doppler_l = 0; 
    % assume LOS -> and target static -> we know the doppler shift at first
    theta_l = pi / 3; % angle of departure and arrival
    b_l = complex(randn(L, 1), randn(L, 1)); % attenuation for each multi-path
    tau_l = 1e-6 * round(rand(L, 1), 2); % time delay for each multi-path
    
    data = randi([0 3], M * Nt * K, 1);

    y = zeros(K, M, Nr);
    ISAC_data = struct;
    x_nk = pskmod(data, 4, pi/4); % one OFDM symbol at one subcarrier carries one QPSK modulated symbol
    x_nk = reshape(x_nk, K, M, Nt);
    y_n = zeros(K, M, Nr);
    h = zeros(K, M, Nr, Nt);
    
    %% signal sampling in frequency domain
    % here, the nth subcarrier and kth OFDM symbol of Nt TX antenans
    % corresponds to -> nth subcarrier and kth OFDM symbol of Nr Rx antennas
    for q = 1:Q
        for k = 1:K
            for n = 1:M
                steering_vec_r = steering_vec_gen(Nt, theta, d, lambda);
                steering_vec_t = steering_vec_gen(Nr, phi, d, lambda);
                x_hat_nk = reshape(x_nk(k, n, :), Nt, 1);
                y(k, n, :) = b_l * exp(-1j * 2*pi * n* fc * tau_l) * exp(1j * 2*pi * Ts * k * f_doppler_l) ...
                    * steering_vec_r * steering_vec_t.' * x_hat_nk;
                h(k, n, :, :) = b_l * exp(-1j * 2*pi * n* fc * tau_l) * exp(1j * 2*pi * Ts * k * f_doppler_l) ...
                    * steering_vec_r * steering_vec_t.';
                y_n(k, n, :) = y(k, n, :) + 0.01 * complex(randn(1, 1, Nr), randn(1, 1, Nr)); 
            end
        end
    end
    y_norm = y / max(abs(y(:)));
    y_norm_n = y_n / max(abs(y_n(:)));
    ISAC_data.y = y;
    ISAC_data.y_norm = y_norm;
    ISAC_data.y_norm_n = y_norm_n;
    ISAC_data.x = x_nk;
    ISAC_data.h = h;
    ISAC_data.channel = struct("time_delay", tau_l, "f_doppler", f_doppler_l, "Tx_steeringangle", phi, "Rx_steeringangle", theta);
    % data save
    path = '../Deep Learning/test_data/ISAC_QPSK_OFDM_' + string(i);
    save(path, "ISAC_data", "data")
end

% %% plot graphs
% y_sub = reshape(y(:, 2, :), 10000*4, 1);
% y_sub_norm = reshape(y_norm(:, 2, :), 10000*4, 1);
% y_sub_norm_n = reshape(y_norm_n(:, 2, :), 10000*4, 1);
% x_sub = reshape(x_nk(:, 2, :), 10000*4, 1);
% 
% scatterplot (y_sub_norm)
% title('normalised y origin')
% scatterplot (y_sub_norm_n)
% title('normalised y with noise')