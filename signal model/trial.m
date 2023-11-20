%% 
clear; clc;
SNR = 10;
delete (gca)
theta = pi/2 * rand;
phi = theta; % steering angles
Nr = 4; % receive antennas
Nt = 1; % transmit antennas
M = 64; % number of subcarrier
K = 20; % 1000 OFDM symbols
fc = 3e9; % assume central frequency is 3 x 10^8 MHz
lambda = 3e8 / fc;
d = lambda / 2;
scs = 20e3;        % Subcarrier spacing in Hz
Fs = scs * M/2; % Sampling rate (1.28e6 Hz)
Ts = 1 / Fs;       % Sample duration in seconds  

L = 1; % assume line of sight -> only one path, no multipath
f_doppler_l = 0; 
% assume LOS -> and target static -> we know the doppler shift at first
b_l = complex(randn(L, 1), randn(L, 1)); % attenuation for each multi-path
tau_l = 1e-9 * round(rand(L, 1), 2); % time delay for each multi-path

data = randi([0 3], M * Nt * K, 1);

y = zeros(K, M, Nr);
x_nk = pskmod(data, 4, pi/4); % one OFDM symbol at one subcarrier carries one QPSK modulated symbol
x_nk = reshape(x_nk, K, M, Nt);
y_n = zeros(K, M, Nr);
h = zeros(K, M, Nr, Nt);

for k = 1:K
    for n = 1:M
        steering_vec_r = steering_vec_gen(Nt, theta, d, lambda);
        steering_vec_t = steering_vec_gen(Nr, phi, d, lambda);
        x_hat_nk = reshape(x_nk(k, n, :), Nt, 1);
        y(k, n, :) = b_l * exp(-1j * 2*pi * n * fc * tau_l) * exp(1j * 2*pi * Ts * k * f_doppler_l) ...
            * steering_vec_r * steering_vec_t.' * x_hat_nk;
        h(k, n, :, :) = b_l * exp(-1j * 2*pi * n* fc * tau_l) * exp(1j * 2*pi * Ts * k * f_doppler_l) ...
            * steering_vec_r * steering_vec_t.';
        P_n = abs(y(1, 1, 1)) / (10^(SNR/10));
        y_n(k, n, :) = y(k, n, :) + sqrt(P_n) * complex(randn(1, 1, Nr), randn(1, 1, Nr));
    end
end
y_n_0 = y_n(:, 1, :);
y_n = reshape(y_n_0, K, Nr).';
% calculate covariance matrix
Rxx= y_n*y_n'/K;
    
    %%
[EV,D]=eig(Rxx);                   % decomposition
EVA=diag(D)';                      
[EVA,I]=sort(EVA);                 
EV=fliplr(EV(:,I));                
derad = pi/180;      % degree -> radians 
cnt = 0;

% precision for angle scanning
aaa = 0;
    
% precision for time_delay scanning
precision = 0.001;
% scan through the time_delays and generate a spectrum
a = steering_vec_gen(Nr, theta, d, lambda);

for itd = 0:precision:1

    aaa = aaa+1;
    time_delay(aaa)=itd;
    td = exp(-1j * 2*pi * 1e-9 * fc * itd) * a;
    En=EV(:, 2:end);                   % take the Nt+1 to Nr element of the matrix to form a subspace
    Pmusic(aaa)=1/(td' * (En * En') * td);

end
Pmusic = abs(Pmusic);
Pmmax = max(Pmusic)';
Pmusic=10*log10(Pmusic/Pmmax);          % logarithm operation

[m, p] = max(Pmusic);
error_6dB(aaa) = abs(p*precision - theta / derad);

%%
h=plot(time_delay,Pmusic, '-o', 'MarkerIndices', p, 'MarkerFaceColor', 'red', 'MarkerSize',15);
% hold on;
% line([tau_l / 1e-8, tau_l / 1e-8], [-40, m], 'linestyle', ':', 'linewidth', 2.5)
set(h,'Linewidth',2);
xlabel('time delay (ns)');
ylabel('Space Spectrum (dB)');
line([tau_l/1e-9, tau_l/1e-9], [-25, m], 'linestyle', ':', 'linewidth', 2.5)
% legend ('MUSIC', 'True Value');
title('MUSIC AOA Scanning')
legend ('MUSIC', 'True Value');
set(gca, 'XTick',0:0.1:1, 'FontSize', 30, 'LineWidth', 1.5);
grid on;