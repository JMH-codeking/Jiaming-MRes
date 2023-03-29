function [X, S] = mimo_ofdm_comm_signal_gen(Q, M, Mt)
% parameters for generating COMM symbol:
    % Q：number of time samples -> over sampling, so Q larger than M
    % M: number of subcarriers
    % Mt: number of transmit antennas
    
    F = exp(1j * 2 * pi / Q * (1:M)' * (1:Q)).'; % sub-carrier matrix -> Q x M
    
    S = ofdm_qpsk_generator(M, Mt);
    
    X = F * S / sqrt(Q);
end