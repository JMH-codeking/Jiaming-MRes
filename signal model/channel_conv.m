function [y_new, t] = channel_conv(t_interval, x_origin, channel_impulse)
% parameters used for creating channel convolution for signal
    % t_interval: sampling interval
    % x_origin: original signal -> a [Q, Mt]
    % channel_impulse: channel impulse response
    for i = 1:Mt
        y = conv(x_origin, channel_impulse);
        
        idx_start = find(y, 1); % conv makes signal start from this point
        idx_end = idx_start + length(x_origin)-1;
    
        y_new = y(idx_start: idx_end);
        t = t_interval*(idx_start-1):t_interval:t_interval*(idx_start-1) + ...
        (length(y_new)-1)*t_interval;
    end
end