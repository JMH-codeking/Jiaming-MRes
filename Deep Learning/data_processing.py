import scipy.io as spio
import numpy as np
from itertools import chain
import torch
import pathlib

def complex2real_fDOM(
    data, # complex data with shape [cnt, carrier number, OFDM symbol number, Nr]
    M,
    Nr,
    symbol_num,
    cnt
):
    '''transform the complex data pack into real LSTM inputs with shape
        [10000 64 4 2]
        -> 64 OFDM carriers, 4 QPSK symbols2 numbers(real and imag) for each symbol
    
        one time -> 64 carriers are input as 64 batches, each batch 4 x 2 matrix
    '''

    lstm_input = np.zeros(shape = (cnt, symbol_num, M, Nr, 2))
    label = np.zeros(shape = (cnt, symbol_num, M, Nr))
    cnt = 0
    for _data in data:
        for i in range(symbol_num):
            for j in range(M):
                complex_ = _data[i][j]
                _real = np.array([])
                _label = np.array([])
                k = 0
                for _complex in complex_:  
                    real_ = np.array([np.real(_complex), np.imag(_complex)])
                    # _real = np.append(_real, real_)
                    # _label = np.append(_label, mapping(real_))
                    lstm_input[cnt, i, j, k, :] = real_
                    label[cnt, i, j, k] = np.array(mapping(real_))
                    k+=1
        cnt += 1
    return torch.tensor(lstm_input, dtype = torch.float32), \
        torch.tensor(label, dtype = torch.long)

def mimo_ofdm_CE (data, label):
    ''' calculate the cross entropy loss for 4-dim MIMO-OFDM symbols
    
    input data is of shape `symbol number x N_ofdm x Nr x 4`
    label is of shape `symbol number x N_ofdm x Nr`

    experiment show it is the same as nn.CrossEntropyloss() after permute
        -> permute is the right choice rather than reshaping
    '''

    channel_num, symbol_num, carrier_num, _ = data.shape
    loss = []
    for i in range (symbol_num):
        for j in range (carrier_num):
            for k, cls in enumerate (label[i, j]):
                x_class = -data[i, j, k][cls]
                log_x_j = torch.log( 
                    sum(
                        [torch.exp(l) for l in data[i, j, k]]
                    )
                )
            loss.append(x_class + log_x_j)



def mapping(
    data
):
    '''input a data sequence [real imag]
    
    output the QPSK modulation scheme label 1234
    '''

    _real = data[0]
    _imag = data[1]
    if _real > 0 and _imag > 0:
        return 0
    if _real > 0 and _imag < 0:
        return 1
    if _real < 0 and _imag > 0:
        return 2
    else:
        return 3

def time_delay_addition_label(
    data,
    time_delay,
    time_interval = 0.01,
    noise_sigma = 0.1,
):
    h = data.shape[0] # actually h is the time samples of the data
    w = data.shape[1]
    l = data.shape[2]
    cnt = int(time_delay / time_interval)
    label = [0] * cnt + [1] * h
    h_new = h + cnt
    _data_timedelay = np.zeros(
        shape = (h_new, w, l)
    )
    for i in range(cnt):
        _data_timedelay[i, :, :] = noise_generator(
            noise_sigma,
            w, l
        )

    _data_timedelay[cnt:, :, :] = data

    return _data_timedelay, label


def ofdm_slicing(
    data,

    channel_num = 4,
    ofdm_carrier_symbol = 32
):
    ''' data reshaping
ï
    data is 2D -> reshape into 3D

    e.g. 4 x 32000 -> 1000 x 1 x 4 x 32
    '''

    size = data.shape[1]
    seq_len = int(size / ofdm_carrier_symbol)

    data_reshape = np.reshape(data, (seq_len, 1, channel_num, ofdm_carrier_symbol))
    return data_reshape


def train_test_valid(
    transmitted_data,
    received_data,
    portion = 0.75, 
    DL_model = 'CNN',
    # assume 0.75 of training data in which 0.25 used for validation
):
    '''define train data, validation data and test data

        1) each complex number is turned into real, before being separated

        2) assume 75% data as train data, in which 25% as validation.
    '''

    _transmitted = complex2real(transmitted_data)
    _received = complex2real(received_data)

    length_traintest = int(np.shape(_transmitted)[0])
    cnt_traintest = int(length_traintest * portion)
    length_valid = int(cnt_traintest * portion)
    
    ''' we know transmitted data and get the received data

    here x is the input -> received data
    here y is the output -> transmitted data
    '''
    x_train = _received[0:cnt_traintest, :, :] # train contains validation dataset
    y_train = _transmitted[0:cnt_traintest, :, :]
    
    x_valid = x_train[length_valid:, :, :]
    y_valid = y_train[length_valid:, :, :]
    
    x_test = _received[cnt_traintest:,  :, :]
    y_test = _transmitted[cnt_traintest:, :, :]

    # change into tensor for model training
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_valid = torch.tensor(x_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    if DL_model == 'CNN':
        x_train = reshape_CNN(x_train)
        y_train = reshape_CNN(y_train)
        x_valid = reshape_CNN(x_valid)
        y_valid = reshape_CNN(y_valid)
        x_test = reshape_CNN(x_test)
        y_test = reshape_CNN(y_test)
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test

def matlab_data_extraction_frequencyDOM(
    data_type, # train or test
    cnt, #the number of files required to be analysed
):
    '''extract matlab frequency domain simulated data in python
    
    extract data from a .mat file and change into dictionary
    '''
    _current_path = str(pathlib.Path(__file__).parent)
    _channel = {}
    _tau = list()
    _fdoppler = list()
    _Txsteering = list()
    _Rxsteering = list()
    _x = list()
    _y = list()
    _y_norm = list()
    h = list()
    if data_type == 'train':
        _parent = _current_path + '/train_data/'
    else:
        _parent = _current_path + '/test_data/'
    for i in range(1, cnt+1):
        path = f'{_parent}/ISAC_QPSK_OFDM_{i}.mat'
        data = spio.loadmat(path)
        _tau.append(
            data['ISAC_data']['channel'][0][0]['time_delay'][0][0][0]
        )
        _fdoppler.append(
            data['ISAC_data']['channel'][0][0]['f_doppler'][0][0][0]
        )
        _Txsteering.append(
            data['ISAC_data']['channel'][0][0]['Tx_steeringangle'][0][0][0]
        )
        _Rxsteering.append(
            data['ISAC_data']['channel'][0][0]['Rx_steeringangle'][0][0][0]
        )
        _x.append(
            data['ISAC_data']['x'][0][0]
        )
        _y.append(
            data['ISAC_data']['y'][0][0]
        )
        _y_norm.append(
            data['ISAC_data']['y_norm'][0][0]
        )
        h.append(
            data['ISAC_data']['h'][0][0]
        )
        
    _channel['time_delay'] = _tau
    _channel['doppler_shift'] = _fdoppler
    _channel['_Txsteering'] = _Txsteering
    _channel['_Rxsteering'] = _Rxsteering

    return _channel, np.array(_x), np.array(_y), np.array(_y_norm), np.array(h)


def matlab_data_extraction_timeDOM(
    transmitted_data_file: str,
    channel: str,
    received_data_file: str,
):
    '''extract data from matlab files
    
    '''
    
    if isinstance(transmitted_data_file, str) == False:
        raise TypeError("false input for transmitted data type, change and re-input")
    if isinstance(channel, str) == False:
        raise TypeError("false input for channel type, change and re-input")

    if isinstance(received_data_file, str) == False:
        raise TypeError("false input for received data type, change and re-input")
    
    transmitted_data = spio.loadmat(transmitted_data_file)
    channel = spio.loadmat(channel)
    received_data = spio.loadmat(received_data_file)

    '''here, extract the s_qpsk from transmitted_data 
    
    extract into raw constellation of qpsk   
    '''

    constellation_data = np.array(transmitted_data['s_qpsk'])
    x_transmitted = np.array(transmitted_data['x_t'])

    channel_matrix_nodoppler = np.array(channel['h_t_withoutdoppler'])
    channel_matrix_withdoppler = np.array(channel['h_t_withdoppler'])

    received_data_no_noise = np.array(received_data['y'])
    received_data_with_noise = np.array(received_data['y_n'])
    received_data_with_noise_no_doppler = \
        np.array(received_data['y_n_nodoppler'])

    '''data mapping:
    
    complex constellation data mapped into real data realm 
        -> the integer real data realm is the processed data
    '''
    mapped_constellation = qpsk_mapping(constellation_data)
    

    return constellation_data, mapped_constellation, x_transmitted, \
        channel_matrix_nodoppler, channel_matrix_withdoppler, \
            received_data_no_noise, received_data_with_noise, \
                received_data_with_noise_no_doppler 
'''
this is the received data we want to deal with 

    -> assume perfect sync & perfect removal of doppler shift based on 
        the object detected is static, so the y we deal with is actually the y with
        noise but assume already removed doppler shift
'''

def qpsk_mapping(
    qpsk_constellation
):  
    '''map the QPSK constellation from complex to real
    
    '''

    mapping = {
        1+1j: 1,
        1-1j: 2,
        -1+1j: 3,
        -1-1j: 4,
    }
    _q = np.array(qpsk_constellation)
    H = _q.shape[0]
    L = _q.shape[1]
    W = _q.shape[2]
    mapped_constellation = _q
    for i in range(H):
        for j in range(L):
            for k in range(W):
                mapped_constellation[i, j, k] = mapping[_q[i, j, k]]
        
    mapped_constellation = np.real(mapped_constellation)
    mapped_constellation = np.array(mapped_constellation)
    return mapped_constellation


def complex2real(complex_data):
    '''change complex signals to real sequence

    e.g. 1+2j -> [1, 2]  
    imput dim: [x, y]
    output dim: [x, 2, y]  
    '''
    
    _list = complex_data
    
    w = _list.shape[0]
    h = _list.shape[1]
    l = _list.shape[2]
    _list = np.zeros((w, h, l * 2))
    i = 0
    for __c in complex_data:
        j = 0
        for _c in __c:
            _real = np.real(_c)
            _complex = np.imag(_c)
            _rc = np.array(list(chain.from_iterable(zip(_real, _complex))))
            _list[i, j, : ] = _rc
            j = j + 1
        i = i + 1

    return np.array(_list)

def reshape_CNN(
        data,
        channel_num = 1,
):
    '''reshape data into data size that CNN can understand

    (1000, 3, 4) -> (1000, 1, 3, 4) because my signal is only one OFDM channel signal
    '''

    H = data.shape[0]
    L = data.shape[1]
    W = data.shape[2]


    return data.reshape(H, channel_num, L, W)

def noise_generator(
    noise_sigma,    
    size,
    is_complex = False,
    is_tensor = True,
    is_CNN = True,
        
):
    '''complex AWGN with general shape of [1, Nt, M] -> 1 channel, Nt x M size

    CNN input is different in output size, which has [1, 1, Nt, M]
    '''

    l = size[0]
    w = size[1]
    # this is the length of complex signal after vectorising, so divide back
    noise = noise_sigma * (
        np.random.randn(1, l, w) + 1j * np.random.randn(1, l, w)
    )
    if is_complex == False: # vectorize the complex signal
        noise = complex2real(noise)

    if is_tensor:
        noise = torch.tensor(noise, dtype = torch.float32)
        if is_CNN:
            noise = noise.reshape(1, 1, noise.shape[1], noise.shape[2])
    return noise


def torch_norm(
    data
):
    ''' normalise a tensor dataset to [-1, 1]
    '''

    _data = data.abs()
    _div = _data.max()
    ans = _data / _div
    ans = ans.type(torch.float32)

    return ans, _div
    
def symbol_acc(
    data, # Nt * M
    symbol_original,
    batch_size = 1,
    rou = 1,
):
    '''symbol calculation & accuracy calculation
    
    X_transmitted = F * S / sqrt(Q -> S = F^-1 * X_transmitted * sqrt(Q)
    '''

    data = symbolise(data, batch_size=batch_size)
    symbol_original = symbolise(symbol_original, batch_size=batch_size)
    # here original data and data have been turned into complex `batch_size x Nt x M` ndarray

    Nt = data.shape[1]
    M = data.shape[2]

    Q = rou * M
    m = np.reshape(np.arange(1, M+1), (1, M))
    # m = torch.expand_dims(m, axis = 0).T # M x 1

    q = np.reshape(np.arange(1, Q+1), (Q, 1))
    # q = np.expand_dims(q, axis = 0) # 1 x Q
    
    ofdm_carrier = np.exp(
        1j * 2 * np.pi / Q * q * m
    )# F: Q x M -> [Q x 1] * [M x 1] -> [Q x M]
    ser_total = 0
    ber_total = 0

    _ofdm_carrier = np.linalg.inv(ofdm_carrier) # F^-1
    for _data, _symbol_original in zip(data,symbol_original):
        symbol = _ofdm_carrier.dot(_data.T) * np.sqrt(Q) # s = F^-1 * X
        symbol = symbol.T # change the matrix back to initialised Nt * M

        # qpsk symbol calculated -> equalise and compare with original data
        symbol = equalisation(symbol)
        ser, ber = check_identical(symbol, _symbol_original)
        ser_total += ser
        ber_total += ber
    ser_av = ser_total / batch_size
    ber_av = ber_total / batch_size

    return ser_av, ber_av

def equalisation(
    signal_array,
    modulation = 'qpsk'
):
    '''equalisation for the complex signals received
    
    typically the signal array is a two dim matrix
    '''

    if not isinstance(signal_array, np.ndarray):
        signal_array = np.array(signal_array)

    h = signal_array.shape[0]
    w = signal_array.shape[1]

    signal_array = signal_array.reshape(h*w, 1)
    
    signal_real = np.real(signal_array)
    signal_imag = np.imag(signal_array)

    signal_real = np.array(
        [1 if _s>0 else -1 for _s in signal_real]
    )
    signal_imag = np.array(
        [1 if _s>0 else -1 for _s in signal_imag]
    )

    signal_array = signal_real + 1j * signal_imag
    signal_array = signal_array.reshape(h, w)
    
    return signal_array


def check_identical(
    A, B
):
    '''This is a metric for calculating SER and BER for two ndarray A and B

    SER first calculated, BER TBC
    A and B are symbolised & equalised matrix with shape [Nt, M]
        -> every element in the matrix means one symbol
    '''

    Nt = A.shape[0]
    M = A.shape[1]
    A = A.reshape(Nt* M, 1)
    B = B.reshape(Nt* M, 1)
    ser = 0
    ber = 0

    for _a, _b in zip(A, B):
        # if not equal, error rate + 1
        _a_real = np.real(_a)
        _b_real = np.real(_b)
        _a_imag = np.imag(_a)
        _b_imag = np.imag(_b)
    
        _a = _a_real + 1j * _a_imag
        _b = _b_real + 1j * _b_imag
        # _a and _b here are complex numbers
        if not np.equal(_a, _b):
            ser += 1
        if not np.equal(_a_real, _b_real):
            ber += 1
        if not np.equal(_a_imag, _b_imag):
            ber += 1

    ser = ser / (M * Nt)
    ber = ber / (M*Nt*2) # two bits one symbol

    return ser, ber

def channel_matrix_ana(
    channel,
    M = 4, # number of antennas
    d = 1,
    fc = 3e11,
):
    '''Analysing the channel matrix and get attenuation & steering angles
    
    the channel is like: [b_l b_l * exp(2j * 2pi * d/lambda * sin(alpha)) ...] 
        -> alpha and b_l wanted
    '''
    from numpy import ndarray
    if isinstance(channel, np.ndarray) == False:
        channel = np.array(channel)

    wavelength = 3e8 / fc

    b_l = channel[0, 0] 
    # alpha is the first matrix element -> get angle from the steering angle
    
    denom = 2j * 2*np.pi * d / wavelength

    alpha = np.arcsin(np.log(channel[1, 1] / b_l) / denom)
    return b_l, alpha

def symbolise(
    data,
    batch_size = 1,
):
    '''turning the matrix that is already vectorized back into real + complex

    input data is tensor with shape [batch size, 1, Nt*2, Mt]
    output data is complex ndarray with shape [Nt, M]
    e.g. tensor [2, 3; 4, 5] -> ndarray [2 + 3j; 4 + 5j]
    '''
    
    data = torch.squeeze(data) # change into [2*Nt x M]
    M = data.shape[1]
    Nt = int(data.shape[0]/2)
    data = torch.reshape(data, (batch_size, Nt*2, M))
    symbolised_matrix = np.zeros((batch_size, Nt, M), dtype=np.complex_) 
        # the array should be batch_size x Nt x M originally
    for k in range (batch_size):
        for j in range (Nt):
            for i in range(M):
                symbolised_matrix[k, j, i] = \
                data[k, j, i].detach().numpy() + 1j * data[k, j+Nt, i].detach().numpy()

    return np.array(symbolised_matrix)

def complex_mse_loss(output, target):
    return (0.5*(output - target)**2).mean(dtype = torch.complex64)

def complex_real_identity(
    received_data,
    channel,
    transmitted_data,
    constellation,
    Nt,
    Nr, 
    M,
    cnt = 0
):
    '''To convert complex values into main domain calculations
    
          - Real(y) -                - Real(H) -Imag(H) -
    y -> |           |         H -> |                    |
          - Imag(y) -                - Imag(H)  Real(H) -
    '''

    t = transmitted_data.shape[0]
    _real_H = np.real(channel)
    _imag_H = np.imag(channel)
    _real_y = np.real(received_data)
    _imag_y = np.imag(received_data)
    _real_x = np.real(transmitted_data)
    _imag_x = np.imag(transmitted_data)
    _real_s = np.real(constellation)
    _imag_s = np.imag(constellation)

    _y = np.zeros(shape = (t, Nr*2, M))
    _s = np.zeros(shape = (t, Nt*2, M))
    _x = np.zeros(shape = (t, Nt*2, M))

    # channel is differently treated when turing into a matrix
    _arr1 = np.column_stack([_real_H, np.negative(_imag_H)])
    _arr2 = np.column_stack([_imag_H, _real_H])
    _H = np.row_stack([_arr1, _arr2]) 
    
    for i in range (t):
        _y[i] = np.row_stack([_real_y[i], _imag_y[i]])
        _x[i] = np.row_stack([_real_x[i], _imag_x[i]])
        _s[i] = np.row_stack([_real_s[i], _imag_s[i]])
    _y = torch.tensor(
        _y,
        dtype = torch.float32
    )
    _H = torch.tensor(
        _H,
        dtype = torch.float32
    )
    _s = torch.tensor(
        _s, 
        dtype=torch.float32
    )
    _x = torch.tensor(
        _x, 
        dtype=torch.float32
    )
    return _y[cnt], _H, _x[cnt], _s[cnt] # choose which time stamp to use


