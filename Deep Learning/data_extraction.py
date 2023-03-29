import scipy.io as spio
import pathlib
import numpy as np
import pandas as pd

def matlab_data_extraction(
    file_name: str,
):
    
    if isinstance(file_name, str) == False:
        raise TypeError("false input for file name type, change and re-input")
    
    signal_data = spio.loadmat(file_name)

    '''here, extract the s_qpsk from signal_data as raw constellation of qpsk
    
    '''

    constellation_data = np.array(signal_data['s_qpsk'])
    y_original = np.array(signal_data['y'])
    y_with_noise = np.array(signal_data['y_n'])
    channel_matrix = np.array(signal_data['h_t'])

    return constellation_data, y_original, y_with_noise, channel_matrix


def qpsk_mapping(
    qpsk_constellation
):
    return
