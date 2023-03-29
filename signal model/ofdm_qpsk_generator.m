function s_gen = ofdm_qpsk_generator( ...
    M, ... % M is the number of sub-carrier -> one sub-carrier carries one qpsk symbol 
    Mt ... % Mt is the number of transmitting antennas
)  

% idea for this is using the randi integer generator of [1, 2], while
% making the generated 2 --> -1 to maintain the principle for QPSK
% modulation, while 1 remain unchanged because 1 is required

    real_part = randi(2, M, Mt);
    imagine_part = randi(2, M, Mt);
    
    real_part(real_part == 2) = -1;
    imagine_part(imagine_part == 2) = -1;
    
    s_gen = real_part + imagine_part * 1j;

end

