import torch
from torch.nn import functional as F
from model.neural_network import *
import pathlib
from data_processing import *
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import hiddenlayer as hl

def train_valid(
    train_data,
    test_data,
    model,
    num_epoch = 10000, 
    _loss = 'cross_entropy',
    save_path = './symbol_detection.pt'
):
    if torch.cuda.is_available():
        device = torch.device('cuda') 
    else:
        device = torch.device('cpu')

    if _loss == "mse":
        loss_function = nn.MSELoss()
    elif _loss == "cross_entropy":
        loss_function = nn.CrossEntropyLoss()
    elif _loss == "complex":
        loss_function = complex_mse_loss
    else:
        assert 'Wrong Name'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    optimizer = torch.optim.Adam (
        model.parameters(),
        lr=0.001,
        weight_decay=0.00001,
    )
    historyl = hl.History()
    canvasl = hl.Canvas()

    valid_num = 0
    for epoch in range (num_epoch):
        
        train_loss_epoch = 0
        valid_loss_epoch = 0

        '''train

        training process -> normal network forward and backward optimisation
        '''
        model.train()
        acc_test = 0
        acc_train = 0
        total_train = 0
        total_test = 0
        train_num = 0
        for step, (x_data, x_label) in enumerate(train_data):
            x_data = x_data.to(device)
            x_label = x_label.to(device)

            out, _ = model(x_data)

            _, predicted_train = torch.max(out,2)
            total_train += x_label.size(0) * x_label.size(1) 
            acc_train += (predicted_train == x_label).sum().item()

            optimizer.zero_grad()
            loss = loss_function(out, x_label)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
            train_num = train_num + 1
        train_loss = train_loss_epoch / train_num
        '''valid
        
        valiation process -> no network forward and backward process
        '''

        model.eval()
        with torch.no_grad():
            for step, (y_data, y_label) in enumerate(test_data):
                y_data = y_data.to(device)
                y_label = y_label.to(device)

                _out, _ = model(y_data)
                _, predicted_test = torch.max(_out,2)
                total_test += y_label.size(0) * y_label.size(1) 
                acc_test += (predicted_test == y_label).sum().item()

        if (epoch+1) % 1 == 0:    
            print (f'-----Epoch {epoch + 1}-----')
            print (f'train_loss: {train_loss: .5f}')
            print (f'train acc: {100*acc_train / total_train :.2f}')
            print (f'test acc: {100*acc_test / total_test :.2f}%' )

        historyl.log(
            epoch, 
            train_loss = train_loss,
            acc_train = 100*acc_train / total_train,
            acc_test = 100*acc_test / total_test
        )
        with canvasl:
            canvasl.draw_plot(
                historyl['train_loss'], 
            )
            canvasl.draw_plot(
                [
                    historyl['acc_train'],
                    historyl['acc_test']
                ]
            )
    torch.save(model.state_dict(), save_path)

    from matplotlib import pyplot as plt
    plt.savefig('./error.png')


def main():
    current_path = str(pathlib.Path(__file__).parent)
    Nt = 4
    Nr = Nt
    M = 16
    symbol_num = 10000

    _channel_train, _x, _y, _y_norm, h = matlab_data_extraction_frequencyDOM('train', 30)
    _channel_test, _X, _Y, _Y_norm, h = matlab_data_extraction_frequencyDOM('test', 30)

    
    '''note that here, _x and _y are sampled in frequency domain 

        -> each Nr for _y corresponds to each Nt for _x
        -> e.g. [1, 1, :] for _x corresponds for [ 1, 1, :] for _y
    '''

    _, _x_label = complex2real_fDOM(_x, M, Nr, symbol_num, 30) 
    # dont need data in x, x is for labelling and classification
    _y_real, _ = complex2real_fDOM(_y_norm, M, Nr, symbol_num, 30) 
    _, _X_label = complex2real_fDOM(_X, M, Nr, symbol_num, 30) 
    # dont need data in x, x is for labelling and classification
    _Y_real, _ = complex2real_fDOM(_Y_norm, M, Nr, symbol_num, 30) 
    # dont need label in y

    '''swap data into carrier-different shape
    
    `channel number x subcarrier number x number of symbols x ...`
    '''

    _x_label = _x_label.permute(2, 0, 1, 3)
    _y_real = _y_real.permute(2, 0, 1, 3, 4)

    _X_label = _X_label.permute(2, 0, 1, 3)
    _Y_real = _Y_real.permute(2, 0, 1, 3, 4)

    

    ofdm_carrier_cnt = 0
    _y_real = _y_real[ofdm_carrier_cnt]
    _x_label = _x_label[ofdm_carrier_cnt]


    _Y_real = _Y_real[ofdm_carrier_cnt]
    _X_label = _X_label[ofdm_carrier_cnt]
    
    train = DataLoader(
        dataset = TensorDataset(
            _y_real[0], 
            _x_label[0]
        ),
        batch_size = 16,
        shuffle = False,
        drop_last=True
    )
    
    cnt_channel = 2
    ofdm_carrier_cnt = 0
    test = DataLoader(
        dataset = TensorDataset(
            _Y_real[1], 
            _X_label[1],
        ),
        batch_size = 16,
        shuffle = False,
        drop_last=True
    )

    from model.transformers import Encoder
    len_traj = 4 
    batch_size = 16
    d_obs = 2
    d_embed = 256 # embedding dimension
    n_heads = 16
    d_k = 16
    d_hidden = 64
    d_class = 4
    n_layers = 6 # Encoder内含
    lstmnet = LSTM_net()
    encoder = Encoder(d_obs, d_embed, d_class, d_k, d_hidden, n_heads, n_layers)
    if torch.cuda.is_available():
        if torch.cuda.device_count()>1:
            encoder = torch.nn.DataParallel(encoder.cuda())
        else:
            encoder = encoder.cuda()


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

    # print(f'shape for random input for H: {_h_input.shape}')

    train_valid(
        train,
        test,
        encoder,
        num_epoch = 10000,
    )

if __name__ == "__main__":
    
    main()