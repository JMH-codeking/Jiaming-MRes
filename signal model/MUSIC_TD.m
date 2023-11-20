delete(gca)
delete(gca)
clc; clear;
% SNR = 10;
for SNR = [2 4 6 8 10]
    theta = pi/2 * rand;
    phi = theta; % steering time_delays
    Nr = 4; % receive antennas
    Nt = 1; % transmit antennas
    M = 16; % number of subcarrier
    K = 2; % 1000 OFDM symbols
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
    b_l= complex(randn(L, 1), randn(L, 1)); 
    

    % time delay for each multi-path
    tau_l = 1e-9 * round (rand(L, 1), 4);
    
    data = randi([0 3], M * Nt * K, 1);
    
    y = zeros(K, M, Nr);
    x_nk = pskmod(data, 4, pi/4); % one OFDM symbol at one subcarrier carries one QPSK modulated symbol
    x_nk = reshape(x_nk, K, M, Nt);
    y_n = zeros(K, M, Nr);
    channel = zeros(K, M, Nr);
    steering_vec_r = steering_vec_gen(Nr, theta, d, lambda);


    steering_vec_t = steering_vec_gen(Nt, phi, d, lambda);

 
    steering_vec_r_n = steering_vec_gen(Nr, theta+0.014, d, lambda);
    for k = 1:K
        for n = 1:M
            x_hat_nk = reshape(x_nk(k, n, :), Nt, 1);
            y(k, n, :) = b_l * exp(-1j * 2*pi * n * fc * tau_l) * exp(1j * 2*pi * Ts * k * f_doppler_l) ...
                * steering_vec_r * steering_vec_t.' * x_hat_nk;
           
            P_n = abs(y(1, 1, 1)) / (10^(SNR/10));

            channel(k, n, :) = b_l * exp(-1j * 2*pi * n* fc * tau_l) * exp(1j * 2*pi * Ts * k * f_doppler_l) ...
                * steering_vec_r + sqrt(P_n) * complex(randn(Nr, 1), randn(Nr, 1));
        
            y_n(k, n, :) = y(k, n, :) + sqrt(P_n) * complex(randn(1, 1, Nr), randn(1, 1, Nr));

            channel_desteering(k, n, :) = reshape (channel(k, n, :), Nr, 1) \ steering_vec_r_n;
        end
    end  
    
    Ry = channel_desteering' * channel_desteering / K;
    %%
    [EV,D]=eig(Ry);                   % decomposition
    EVA=diag(D)';                      
    [EVA,I]=sort(EVA);                 
    EV=fliplr(EV(:,I));                           
    
    aaa = 0;
    
    % precision for time_delay scanning
    precision = 0.001;
    % scan through the time_delays and generate a spectrum
    a = steering_vec_gen(Nr, theta, d, lambda);
   
    for itd = 0:precision:1
    
        aaa = aaa+1;
        time_delay(aaa)=itd;
        num = 1:16;
        td = exp(-1j * 2*pi * num * fc * 1e-9 * itd).';
        En=EV(:, 2:end);                   % take the Nt+1 to Nr element of the matrix to form a subspace
        Pmusic(aaa)=1/(td' * (En * En') * td);
    
    end
    
    Pmusic = abs(Pmusic);
    Pmmax = max(Pmusic)';
    Pmusic=10*log10(Pmusic / Pmmax);          % logarithm operation
    
    [m, p] = max(Pmusic);
    pp = p * precision * 1e-9;
    error = abs(pp - tau_l);
    disp(error * 1e9)
end
%% basic plot
% error_6dB = mean(error_dB);
subplot(2,1,1)
h=plot(time_delay,Pmusic, '-o', 'MarkerIndices', p, 'MarkerFaceColor', 'red', 'MarkerSize',15);
hold on;
% line([tau_l / 1e-8, tau_l / 1e-8], [-40, m], 'linestyle', ':', 'linewidth', 2.5)
set(h,'Linewidth',2);
xlabel('time delay (ns)');
ylabel('Space Spectrum (dB)');
line([tau_l/1e-9, tau_l/1e-9], [-30, m], 'linestyle', ':', 'linewidth', 2.5)
% legend ('MUSIC', 'True Value');
% title('MUSIC AOA Scanning')
legend ('MUSIC', 'True Value');
set(gca, 'XTick',0:0.1:1, 'FontSize', 30, 'LineWidth', 1.5);
grid on;
f = gca;
% exportgraphics(f, '../../Paper Writing/MUSIC_td_finding.png', 'Resolution', 300);
%% plot
error = [0.0014, 0.0012, 4e-4, 6e-4, 0.0012];

SNR = 2:2:10;

subplot(2,1,2)
gca_aoa = semilogy(SNR, error, 's-', 'LineWidth', 3);

ylim([1e-4 1])
set(gca_aoa, 'Linewidth', 2);
xlabel('SNR value (dB)');
ylabel('Error (ns)');
% title('TD Estimation Evaluation');
set (gca,'XTick', 2:2:10, 'FontSize', 30, 'LineWidth', 1.5)
grid on;
f = gca;
% exportgraphics(f, '../../Paper Writing/MUSIC_td_evaluation.png', 'Resolution', 300);
