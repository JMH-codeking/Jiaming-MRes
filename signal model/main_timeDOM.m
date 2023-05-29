%% signal parameter initialisation
clc; clear;
rou = 1; % oversampling factor
M = 64; % number of subcarriers
Q = rou * M; % number of time samples
fc = 3e11; % assume central frequency is 3 x 10^8 MHz
d = 0.5 * 3e8/3e11; % interval of antennas is half a wavelength
t = 0:0.01:10-0.01;     

%% channel signal initialisation
Mr = 16; % 16 antennas in the receiver side
Mt = Mr; % assume transmitter side and receiver side are identical
L = 1; % assume line of sight -> only one path, no multipath
f_doppler_l = randi(10000, L, 1); 
% assume LOS -> and target static -> we know the doppler shift at first
theta_l = pi / 3; % angle of departure and arrival
b_l = complex(randn(L, 1), randn(L, 1)); % attenuation for each multi-path
tau_l = round(rand(L, 1), 2); % time delay for each multi-path

%% generation of final signal model
[x_t, y, y_n, y_n_nodoppler, s_qpsk, h_t_withoutdoppler, h_t_withdoppler] = ... 
  signal_generation(t, rou, M, fc, d, Mt, f_doppler_l, theta_l, b_l, tau_l);

%% data save
save('../Deep Learning/data/transmitted_data', "s_qpsk", "x_t")
save('../Deep Learning/data/channel', "h_t_withoutdoppler", "h_t_withdoppler")
save('../Deep Learning/data/received_data', 'y', 'y_n', 'y_n_nodoppler')