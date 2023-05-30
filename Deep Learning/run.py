import torch
from torch.nn import functional as F
from model.neural_network import *
import pathlib
from data_processing import *
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import hiddenlayer as hl

def train_valid(
    received_data,
    channel_data,
    transmitted_constellation,
    Nt,
    Nr,
    M,
    signal_net = None,
    channel_net = None, 
    batch_size_train = 1,
    num_epoch = 1000, 
    _loss = 'mse',
    save_path_channel = './model/channelNet_ISAC.pt',
    save_path_signal = './model/signalNet_ISAC.pt',
):
    if _loss == "mse":
        loss_function = nn.MSELoss()
    elif _loss == "cross_entropy":
        loss_function = nn.CrossEntropyLoss()
    elif _loss == "complex":
        loss_function = complex_mse_loss
    else:
        assert 'Wrong Name'
    


    channel_net = Channel_ana()

    optimizer = torch.optim.Adam(
        [
            {'params': signal_net.parameters(), 'lr': 1e4}, 
            {'params': channel_net.parameters(), 'lr': 1e-4}
        ], 
        lr=0.01,
        weight_decay=0.001,
        )
    
    historyl = hl.History()
    canvasl = hl.Canvas()

    _real_x = torch.randn(Nt, M)
    _imag_x = torch.randn(Nt, M)

    _x_input = torch.cat((_real_x, _imag_x), 0)
    _x_input = _x_input.reshape(1, 1, Nt*2, M)
    # print(f'shape for random input for x: {_x_input.shape}')

    _real_h = torch.randn(Nr, Nt)
    _imag_h = torch.randn(Nr, Nt)

    _h_input = torch.cat((
        torch.cat(
            (_real_h, torch.negative(_imag_h)),
            1
        ), # first row for h
        torch.cat(
            (_imag_h, _real_h),
            1
        ), # second row for h
    ), 0)
    _h_input = _h_input.reshape(1, 1, Nr*2, Nt*2)

    _x_input = _x_input.detach().clone()
    # H_initial_input = H_initial.detach().clone()


    train_num = 0

    for epoch in range (num_epoch):
        print (f'-----Epoch {epoch + 1}-----')
        train_loss_epoch = 0
        valid_loss_epoch = 0

        '''train

        training process -> normal network forward and backward optimisation
        '''
        signal_net.train()
        channel_net.train()
        optimizer.zero_grad()

        # regularize X
        _x_input_constrained = _x_input + \
            0.01 * torch.zeros(
                _x_input.shape, dtype = torch.float32
            ).normal_()
        
        x = signal_net(_x_input_constrained)
        h = channel_net(_h_input)
        
        y = torch.tensordot(h, x)

        loss = loss_function(y, received_data)
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss
        train_num += 1
        
        train_loss = train_loss_epoch / train_num
        print ('train_loss:',train_loss.detach())

        '''valid
        
        valiation process -> no network forward and backward process
        '''

        signal_net.eval()
        channel_net.eval()
        h = channel_net(_h_input)

        x = signal_net(_x_input)

        ser, ber = symbol_acc(
            x, transmitted_constellation, batch_size=batch_size_train
        )
        channel_error = loss_function(h.squeeze_(), channel_data)

        print(f'channel error is: {channel_error.detach()}')
        print (f'SER is {ser}\nBER is {ber}')

        # historyl.log(
        #     epoch, 
        #     train_loss = train_loss,
        # )
        # with canvasl:
        #     canvasl.draw_plot(
        #     [
        #         historyl['train_loss'], 
        #     ]
        # )
        h = h.detach()
        print(f'recovered attenuation is:{h[0, 0]} original attenuation is: {channel_data.squeeze_()[0, 0]}')
    torch.save(channel_net.state_dict(), save_path_channel)
    torch.save(signal_net.state_dict(), save_path_signal)

    from matplotlib import pyplot as plt
    plt.savefig('./error.png')


def main():
    current_path = str(pathlib.Path(__file__).parent)
    transmitted_data = f'{current_path}/train_data/ISAC_QPSK_OFDM_1.mat'
    channel = f'{current_path}/data/channel.mat'
    received_data = f'{current_path}/data/received_data.mat'
    Nt = 4
    Nr = Nt
    M = 64
    symbol_num = 100

    _channel, _x, _y = matlab_data_extraction_frequencyDOM('train', 10)
    '''note that here, _x and _y are sampled in frequency domain 

        -> each Nr for _y corresponds to each Nt for _x
        -> e.g. [1, 1, :] for _x corresponds for [ 1, 1, :] for _y
    '''

    _x_real = complex2real_fDOM(_x, M, Nr, symbol_num)
    _y_real = complex2real_fDOM(_y, M, Nr, symbol_num)

    _x = torch.tensor(_x, dtype=torch.cfloat)
    _y = torch.tensor(_y, dtype=torch.cfloat)

    net = LSTM_net()
    
    _x_real = torch.tensor(_x_real, dtype = torch.float32)
    _y_real = torch.tensor(_y_real, dtype = torch.float32)

    print (net(_y_real[0, 0]).shape)

    # constellation_data, mapped_constellation, x_transmitted, channel_matrix_nodoppler, \
    #     channel_matrix_withdoppler, received_data_no_noise, \
    #         received_data_with_noise, received_data_with_noise_no_doppler \
    #         = matlab_data_extraction(transmitted_data, channel, received_data)
    
    # test code for channel_ana
    # b_l, alpha = channel_matrix_ana(channel_matrix_nodoppler)
    # print(f'attenuation is: {b_l}, steering angle is: {alpha}')

    # _y, _H, _x, _s = complex_real_identity(
    #     received_data_with_noise_no_doppler,
    #     channel_matrix_nodoppler,
    #     x_transmitted,
    #     constellation_data,
    #     Nt,
    #     Nt,
    #     M,
    #     cnt= 10,
    # )

    # _s = _s.reshape(1, 1, Nt*2, M)
    # _x = _x.reshape(1, 1, Nr*2, M)
    # _H = _H.reshape(1, 1, Nr*2, Nt*2)
    
    '''model training
    
    '''

    # print(f'shape for random input for H: {_h_input.shape}')

    # train_valid(
    #     received_data = _y,
    #     # channel_data,
    #     transmitted_constellation = _s,
    #     channel_data = _H,
    #     Nt = Nt,
    #     Nr = Nr,
    #     M = M,
    #     num_epoch = 10000,
    # )

if __name__ == "__main__":
    
    main()