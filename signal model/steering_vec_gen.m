function steering_vec = steering_vec_gen(Mt, theta, d, lambda)
% parameters for create steering vectors: 
    % Mt:      the number of antennas 
    % theta:  the angle of departure / arrival 
    % d:      the interval of antennas
    % lambda: the wavelength

    cnt = 0:Mt-1;
    steering_vec = exp(1j * cnt' * 2*pi * d / lambda .* sin(theta));
end