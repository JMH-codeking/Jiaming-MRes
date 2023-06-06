%% first thing
% Parameters
numTx = 2; % Number of transmit antennas
numRx = 12; % Number of receive antennas
numPaths = 1; % Number of paths
pathDelays = 1e-6; % Delay for each path
avgPathGains = -10; % Average gain for each path in dB

% Create MIMO channel object
mimoChan = comm.MIMOChannel( ...
    'SampleRate', 1e6, ... % Sample rate
    'PathDelays', pathDelays, ...
    'AveragePathGains', avgPathGains, ...
    "SpatialCorrelationSpecification",  "None", ...
    'NumReceiveAntennas', numRx, ...
    'FadingDistribution', 'Rayleigh', ...
    'NumTransmitAntennas', numTx, ...
    'RandomStream', 'mt19937ar with seed', ...
    'Seed', 22);

% Create random symbols to send
x = randi([0 1], 1000, numTx);

% Pass symbols through channel
y = mimoChan(x);

% y now contains the received symbols with channel effects
% pathGains contains the actual path gains used by the channel

%% second simulation
clc; clear;
% Channel parameters
numTaps = 3;  % Number of taps
numTxAntennas = 4;  % Number of transmit antennas
numRxAntennas = 4;  % Number of receive antennas

% Generate random phase, gain, and delay values for each tap and antenna
phases = rand(numTaps, numRxAntennas);
gains = rand(numTaps, numRxAntennas);
delays = rand(1, numTaps);

% Impulse response initialization
impulseResponse = zeros(numTaps, numTxAntennas, numRxAntennas);

% Generate impulse response for each tap, transmit antenna, and receive antenna
for tap = 1:numTaps
    for txAntenna = 1:numTxAntennas
        for rxAntenna = 1:numRxAntennas
            impulseResponse(tap, txAntenna, rxAntenna) = gains(tap, rxAntenna) ...
                * exp(1i * phases(tap, rxAntenna)) ...
                * (tap-1) * delays(tap);  % Delay scaling
        end
    end
end

% Display the impulse response
disp('Impulse Response:');
disp(impulseResponse);

%% pulse shaping
clc; clear;
% Define input parameters
inputSignal = [1 0 1 1 0 1];  % Example input signal
pulseShape = [0.25 0.5 1 0.5 0.25];  % Example pulse shaping filter (raised cosine)
delays = 1e-6;  % Example delays for each tap

% Apply pulse shaping with delays
shapedSignal = pulseShapingWithDelays(inputSignal, pulseShape, delays);

% Display the results
disp('Input Signal:');
disp(inputSignal);

disp('Shaped Signal:');
disp(shapedSignal);

%% pulse shape evaluated at time t
clc; clear;
t = 0.2;   % Time
T = 1;     % Symbol duration

pulseValue = evaluatePulseShape(t, T);
disp('Pulse Shape at t = 0.2:');
disp(pulseValue);



%% functions
function pulseValue = evaluatePulseShape(t, T)
    if abs(t) <= T
        pulseValue = 0.5*(cos(pi*t/T) + 1);
    else
        pulseValue = 0;
    end
end

function shapedSignal = pulseShapingWithDelays(inputSignal, pulseShape, delays)
    % Input:
    %   - inputSignal: Input signal to be shaped
    %   - pulseShape: Pulse shaping filter coefficients (e.g., raised cosine)
    %   - delays: Array of delays for each tap (in number of samples)
    %
    % Output:
    %   - shapedSignal: Output signal after pulse shaping with delays

    % Determine the length of the shaped signal based on the input signal and delays
    shapedLength = length(inputSignal) + max(delays);

    % Initialize the shaped signal with zeros
    shapedSignal = zeros(1, shapedLength);

    % Apply pulse shaping with delays
    for i = 1:length(inputSignal)
        for j = 1:length(pulseShape)
            shapedSignal(i + delays) = shapedSignal(i + delays) + inputSignal(i) * pulseShape(j);
        end
    end
end
