function y = impulse(t, t_l)
    % x is a vector
    % We create an output vector of only 0 (our default value)
    y = zeros(1, length(t));
    % We find indexes of input values equal to 0,
    % and make them 1
    for i = 1:length(t_l)
        y(abs(t - t_l(i)) <= 0.0001) = 1;
    end
end