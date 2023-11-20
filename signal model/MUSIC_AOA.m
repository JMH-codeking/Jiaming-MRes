clear; clc;

delete (gca)
delete (gca)

theta_1 = pi/4 *rand;
phi = theta_1; % steering time_delays
SNR = 2;
Nr = 4; % receive antennas
Nt = 1; % transmit antennas
M = 20; % number of subcarrier
K = 20; % 1000 OFDM symbols
% L = 3; % number of multi-paths
fc = 3e8; % assume central frequency is 3 x 10^8 MHz
lambda = 3e8 / fc;
d = lambda / 2;
scs = 20e3;        % Subcarrier spacing in Hz
Fs = scs * M/2; % Sampling rate (1.28e6 Hz)
Ts = 1 / Fs;       % Sample duration in seconds  

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
steering_vec_r = steering_vec_gen(Nr, theta_1, d, lambda);


steering_vec_t = steering_vec_gen(Nt, phi, d, lambda);

for k = 1:K
    for n = 1:M
        x_hat_nk = reshape(x_nk(k, n, :), Nt, 1);
        y(k, n, :) = b_l * exp(-1j * 2*pi * n * fc * tau_l) * exp(1j * 2*pi * Ts * k * f_doppler_l) ...
            * steering_vec_r  * x_hat_nk;
     
        P_n = abs(y(1, 1, 1)) / (10^(SNR/10));
        y_n(k, n, :) = y(k, n, :) + sqrt(P_n) * complex(randn(1, 1, Nr), randn(1, 1, Nr));
    end
end
 
y_n = reshape(y_n, M*K, Nr).';
% calculate covariance matrix
Rxx= y_n*y_n'/K;

[EV,D]=eig(Rxx);                   % decomposition
% EVA=diag(D)';                      
% [EVA,I]=sort(EVA);                 
EV=fliplr(EV);                
derad = pi/180;      % degree -> radians 
cnt = 0;

% precision for angle scanning
precision = 0.01;
% scan through the angles and generate a spectrum
for iang = 0:precision:90
    cnt = cnt+1;
    angle(cnt)=iang;
    phim=derad*angle(cnt);
    a=steering_vec_gen(Nr, phim, d, lambda); 
    En=EV(:,Nt+1:Nr);                   % take the Nt+1 to Nr element of the matrix to form a subspace
    Pmusic(cnt)=1/(a'* (En*En') *a);
end
Pmusic = abs(Pmusic);
Pmmax = max(Pmusic)';
Pmusic=10*log10(Pmusic/Pmmax);          % logarithm operation

[m, p] = max(Pmusic);
error_6dB = abs(p*precision - theta_1 / derad);


disp(theta_1)
disp(p*precision * derad)
subplot(2,1,1)
h=plot(angle,Pmusic, '-o', 'MarkerIndices', p, 'MarkerFaceColor', 'red', 'MarkerSize',15);
hold on;
plot (angle, Pmusic)
hold on;
line([p*precision, p*precision], [-45, m], 'linestyle', ':', 'linewidth', 2.5)
set(h,'Linewidth',2); 
xlabel('Angle of Arrival (degree)');
ylabel('Space Spectrum (dB)');
legend ('MUSIC', 'True Value');
% title('MUSIC AOA Scanning')
set(gca, 'XTick',0:10:90, 'FontSize', 30, 'LineWidth', 1.5);
% f_finding = gca;
grid on;
% disp(mean(abs(error_6dB)))
% exportgraphics(f_finding, '../../Paper Writing/MUSIC_aoa_finding.png', 'Resolution', 300);
%% plot of various dB values for AOA estimation
% delete (gca)
error = [0.24, 0.18, 0.11, 0.031, 0.0165];

SNR = [2, 4, 6, 8, 10];

subplot(2,1,2)
gca_aoa = plot(SNR, error, 's-', 'LineWidth', 3);

set(gca_aoa, 'Linewidth', 2);
xlabel('SNR value (dB)');
ylabel('Error (degrees)');
% title('AOA Estimation Evaluation');
set (gca,'XTick', 2:2:10, 'FontSize', 30, 'LineWidth', 2)
grid on;
f = gca;
% exportgraphics(f, '../../Paper Writing/MUSIC_aoa_evaluation.png', 'Resolution', 300);

