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

    mapped_constellation = qpsk_mapping(constellation_data)

    return mapped_constellation, y_original, y_with_noise, channel_matrix


def qpsk_mapping(
    qpsk_constellation
):  
    mapping = {
        1+1j: 1,
        1-1j: 2,
        -1+1j: 3,
        -1-1j: 4,
    }
    mapped_constellation = list()
    for _constellation in qpsk_constellation:
        _constellation = [mapping[__c] for __c in _constellation]
        mapped_constellation.append(_constellation)

    mapped_constellation = np.array(mapped_constellation)