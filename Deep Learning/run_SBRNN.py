import torch
import numpy as np
from data_processing import *
import scipy.io as spio
from torch.nn import functional as F


Nr = 4
carrier_num = 16
symbol_num = 200
channel_num = 4000

def cross_entropy(y_true,y_pred):
    C=0
    # one-hot encoding
    for col in range(y_true.shape[-1]):
        y_pred[col] = y_pred[col] if y_pred[col] < 1 else 0.99999
        y_pred[col] = y_pred[col] if y_pred[col] > 0 else 0.00001
        C+=y_true[col]*torch.log(y_pred[col])+(1-y_true[col])*torch.log(1-y_pred[col])
    return -C
_channel = {}
_tau = list()
_fdoppler = list()
_Txsteering = list()
_Rxsteering = list()
_x = list()
_y = list()
_y_norm = list()
h = list()

for i in range(1, channel_num+1):
    path = f'./test_data/ISAC_QPSK_OFDM_{i}.mat'
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

x, _, y_norm, h = np.array(_x), np.array(_y), np.array(_y_norm), np.array(h)
print (x.shape)
print (y_norm.shape)
def mapping_qpsk (
    data
):
    _real = np.real(data)
    _imag = np.imag(data)

    if _real > 0 and _imag > 0:
        return 0
    if _real > 0 and _imag < 0:
        return 1
    if _real < 0 and _imag > 0:
        return 2
    else:
        return 3
# take the first carrier as an example

y_lstm_isac = np.zeros(shape = (carrier_num, channel_num, symbol_num, Nr, 2))
label_isac = np.zeros(shape = (carrier_num, channel_num, symbol_num))
for n in range (carrier_num):
    for i in range (channel_num):
        for j in range (symbol_num):
            label_isac[n, i, j] = mapping_qpsk (
                x[i, j, n]
            )
            for k in range (Nr):
                y_lstm_isac[n, i, j, k] = np.array(
                    [
                        np.real(y_norm[i, j, n, k]),
                        np.imag(y_norm[i, j, n, k]),
                    ]
                )

y_carrier0, label_carrier0 = y_lstm_isac[0], label_isac[0]

y_train = y_carrier0
label_train = label_carrier0

print (f'training data shape: {y_train.shape}')
print (f'training label shape{label_train.shape}')

import torch
from torch import nn

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size, hidden_size, bidirectional=True, num_layers = 3
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes) # times 2 because of bidirection

    def forward(self, x):
        seq_len, batch_size, _ = x.size()
        h0 = torch.zeros(2*self.lstm.num_layers, batch_size, self.hidden_size).to(x.device) # 2 for bidirection 
        c0 = torch.zeros(2*self.lstm.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out

# Parameters
input_size = 2 # input size for each LSTM cell (complex numbers)
hidden_size = 20 # hidden state size for LSTM cell
num_classes = 4 # number of classes for classification

# Instantiate the model
model = BiLSTM(input_size, hidden_size, num_classes)

y_train = torch.tensor(
    y_train,
    dtype = torch.float32
)

label_train = torch.tensor(
    label_train,
    dtype = torch.long
)

optimiser = torch.optim.Adam(
    model.parameters(),
    lr = 0.001,
    weight_decay = 0.0001,
)
loss = nn.CrossEntropyLoss()
import hiddenlayer as hl

canvasl = hl.Canvas()
historyl = hl.History()
window_size = 40
train_cnt = 0
acc_train_list = list()

for epoch in range (100):

    print (f'For Epoch {epoch+1}: ------------------')
    for _y_train, _label_train in zip(
        y_train, label_train
    ):
        train_cnt += 1
        acc_train = 0
        acc_num_train = 0

        for cnt in range (symbol_num - window_size + 1):
            model.train() # train with y_train
            X = _y_train[cnt:cnt + window_size]
            Y = _label_train[cnt: cnt + window_size]
            out = model(X)

            # acc for each sample
            _, predicted = torch.max(out, 1)
            acc_train = acc_train + (predicted == Y).sum().item()
            acc_num_train += window_size

            # backward propagation
            optimiser.zero_grad()
            _loss = loss(out, Y)
            _loss.backward()
            optimiser.step()

        acc_train_list.append(acc_train / acc_num_train)

        '''record acc for train and test        
        '''

        acc_train_av = sum(acc_train_list) / len(acc_train_list)
        print (f'    - - training step {train_cnt} - -')
        print (f'   average accuracy for training is: {acc_train_av:.2f}')
        historyl.log(
            train_cnt,
            acc_train_average = acc_train_av
        )

        with canvasl:
            canvasl.draw_plot(
                historyl['acc_train_average']
            )

    from matplotlib import pyplot as plt
    plt.savefig(f'./epoch{epoch}.png')
torch.save(model.state_dict(), './symbol_detection.pt')