% Define the received signals
Fs = 1000;  % Sampling rate in Hz
t = (0:1/Fs:1-1/Fs);  % Time vector
f1 = 10;  % Frequency of signal 1 in Hz
f2 = 15;  % Frequency of signal 2 in Hz
A1 = 1;  % Amplitude of signal 1
A2 = 0.5;  % Amplitude of signal 2
phase_delay = 0.2;  % Phase delay in seconds

signal1 = A1 * sin(2*pi*f1*t);  % Signal 1: Sine wave at frequency f1
signal2 = A2 * sin(2*pi*f2*(t - phase_delay));  % Signal 2: Sine wave at frequency f2 with a phase delay

% Add noise to the signals
SNR_dB = 10;  % Signal-to-Noise Ratio in dB
noise_power = 10^(-SNR_dB/10);  % Calculate noise power
noise1 = sqrt(noise_power) * randn(size(signal1));  % Generate noise for signal 1
noise2 = sqrt(noise_power) * randn(size(signal2));  % Generate noise for signal 2
signal1 = signal1 + noise1;  % Add noise to signal 1
signal2 = signal2 + noise2;  % Add noise to signal 2

% Perform time delay estimation
[corr, lags] = xcorr(signal1, signal2);  % Cross-correlation between signals
[max_corr, max_index] = max(abs(corr));  % Find the maximum correlation
estimated_delay = lags(max_index) / Fs;  % Convert the delay to time units

% Perform time delay compensation
time_delay_samples = round(estimated_delay * Fs);  % Convert the delay to samples
compensated_signal2 = circshift(signal2, -time_delay_samples);  % Shift signal2

% Perform further processing on the decorrelated signals if desired
% ...

% Plot the original and decorrelated signals for visualization
subplot(2,1,1);
plot(t, signal1, 'b');
hold on;
plot(t, signal2, 'r');
hold off;
title('Original Signals');
legend('Signal 1', 'Signal 2');
xlabel('Time (seconds)');

subplot(2,1,2);
plot(t, signal1, 'b');
hold on;
plot(t, compensated_signal2, 'r');
hold off;
title('Decorrelated Signals');
legend('Signal 1', 'Compensated Signal 2');
xlabel('Time (seconds)');
