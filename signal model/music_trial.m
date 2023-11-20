% OFDM system parameters
numSubcarriers = 64;        % Number of subcarriers
cpLength = 16;              % Length of cyclic prefix (CP)
symbolLength = numSubcarriers + cpLength; % Total symbol length

% Generate random OFDM symbol
data = randi([0 1], numSubcarriers, 1);      % Random data bits
modulatedData = qammod(data, 64, 'bin');     % QAM modulation
ofdmSymbol = ifft(modulatedData);            % Perform IFFT
ofdmSymbol = [ofdmSymbol(end-cpLength+1:end); ofdmSymbol];  % Add cyclic prefix

% Apply time delay to the OFDM symbol
timeDelay = 5;               % Time delay in samples
delayedSymbol = circshift(ofdmSymbol, timeDelay);

% Apply the MUSIC algorithm for time delay estimation
correlationMatrix = toeplitz(delayedSymbol([1 end:-1:2]), delayedSymbol([1 end:-1:2]));
[U, ~, ~] = svd(correlationMatrix);
noiseSubspace = U(:, numSubcarriers+1:end);
spectrum = sum(abs(conj(noiseSubspace.' * fft(delayedSymbol))).^2, 2);

% Estimate time delay using MUSIC algorithm
[~, delayIndex] = max(spectrum);
estimatedDelay = delayIndex - 1;

% Display estimated time delay
disp(['Estimated Time Delay: ' num2str(estimatedDelay) ' samples']);

% Plot the original and delayed OFDM symbols for visualization
subplot(2, 1, 1);
plot(0:symbolLength-1, abs(ofdmSymbol));
title('Original OFDM Symbol');
xlabel('Time (samples)');
ylabel('Magnitude');

subplot(2, 1, 2);
plot(0:symbolLength-1, abs(delayedSymbol));
title('Delayed OFDM Symbol');
xlabel('Time (samples)');
ylabel('Magnitude');
