clc; clear;

%% signal initialisation
snr = [2, 4, 6, 8, 10];
for SNR = snr
    for i = 1:5
        Q = 1;
        theta = pi/3;
        phi = theta; % steering time_delays
        Nr = 4; % receive antennas
        Nt = 1; % transmit antennas
        M = 2; % number of subcarrier
        K = 20000; % 1000 OFDM symbols
        % L = 3; % number of multi-paths
        fc = 3e8; % assume central frequency is 3 x 10^8 MHz
        lambda = 3e8 / fc;
        d = lambda / 2;
        scs = 20e3;        % Subcarrier spacing in Hz
        Fs = scs * M/2; % Sampling rate (1.28e6 Hz)
        Ts = 1 / Fs;       % Smple duration in seconds  
        
        L = 1; % number of multipath
        f_doppler_l = 0; 
        % assume LOS -> and target static -> we know the doppler shift at first
        
        
        % attenuation for each multi-path
        b_l = complex(randn(L, 1), randn(L, 1)); 
        
    
        % time delay for each multi-path
        tau_l = 1e-9 * round (rand(L, 1), 4);
        
        data = randi([0 3], M * Nt * K, 1);
        
        y = zeros(K, M, Nr);
        x_nk = pskmod(data, 4, pi/4); % one OFDM symbol at one subcarrier carries one QPSK modulated symbol
        x_nk = reshape(x_nk, K, M, Nt);
        y_n = zeros(K, M, Nr);
        channel = zeros(K, M, Nr);
        steering_vec_t = steering_vec_gen(Nt, theta, d, lambda);
        steering_vec_r = steering_vec_gen(Nr, phi, d, lambda);

        ISAC_data = struct;
        
        %% signal sampling in frequency domain
        % here, the nth subcarrier and kth OFDM symbol of Nt Tx antenans
        % corresponds to -> nth subcarrier and kth OFDM symbol of Nr Rx antennas
        for q = 1:Q
            for k = 1:K
                for n = 1:M
                    x_hat_nk = reshape(x_nk(k, n, :), Nt, 1);
                   y(k, n, :) = b_l * exp(-1j * 2*pi * n * fc * tau_l) * exp(1j * 2*pi * Ts * k * f_doppler_l) ...
                        * steering_vec_r * steering_vec_t.' * x_hat_nk;
                    h(k, n, :, :) = b_l * exp(-1j * 2*pi * n* fc * tau_l) * exp(1j * 2*pi * Ts * k * f_doppler_l) ...
                        * steering_vec_r * steering_vec_t.';
                    P_n = abs(y(1, 1, 1)) / (10^(SNR/10));
                    y_n(k, n, :) = y(k, n, L) + sqrt(P_n) * complex(randn(Nr, 1), randn(Nr, 1));
                end
            end
        end
        y_norm = y / max(abs(y_n(:)));
        y_norm_n = y_n / max(abs(y_n(:))); 
            % here normalisation must divide by the same factor for deep
            % learning as normalisation for y_norm is not assumed to be known
            % and denormalising must be done by multiplying the normalisation
            % factor by y_n
        ISAC_data.y = y;
        ISAC_data.y_n = y_n;
        ISAC_data.y_norm = y_norm;
        ISAC_data.y_norm_n = y_norm_n;
        ISAC_data.x = x_nk;
        ISAC_data.h = h;
        % ISAC_data.channel = struct("attenuation", b_l, "time_delay", tau_l, "f_doppler", f_doppler_l, "Tx_steeringangle", phi, "Rx_steeringangle", theta);
        % data save
        path = '../Deep Learning/SIMO_data/' + string(SNR) + 'dBTrial/ISAC_QPSK_OFDM_' + string(i);
        
        save(path, "ISAC_data", "data")
    end
end
%% plot graphs
% y_sub = reshape(y(:, 2, :), 1000*4, 1);
% y_sub = reshape(y(:, 2, :), 1000*4, 1);
% y_sub_n = reshape(y_n(:, 2, :), 1000*4, 1);
% x_sub = reshape(x_nk(:, 2, :), 1000, 1);
% 
% scatter(real(x_sub), imag(x_sub), 60, 'k', 'filled')
% title('Constellation for QPSK', 'FontSize', 30)
% xlabel('Real', 'FontSize', 30)
% ylabel('Imag', 'FontSize', 30)
% %%
% scatter(real(y_sub), imag(y_sub), 60, 'k', 'filled')
% title('received data without noise', 'FontSize', 30)
% xlabel('Real', 'FontSize', 30)
% ylabel('Imag', 'FontSize', 30)
% figure;
% scatter(real(y_sub_n), imag(y_sub_n), 60, 'k', 'filled')
% title('received data with noise', 'FontSize', 30)
% xlabel('Real', 'FontSize', 30)
% ylabel('Imag', 'FontSize', 30)