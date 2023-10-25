import torch
import numpy as np
from data_processing import *
import scipy.io as spio
import torch
import numpy as np
len_traj = 40
batch_size = 4
d_obj = 2
d_embed = 256 # embedding dimension
n_heads = 16
d_k = 16
d_hidden = 64
d_class = 4
n_layers = 6 # Encoder内含

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
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

Nr = 4
carrier_num = 16
symbol_num = 100
channel_num = 1000

_channel = {}
_tau = list()
_attenuation = list()
_fdoppler = list()
_Txsteering = list()
_Rxsteering = list()
_x = list()
_y = list()
_y_norm = list()
_y_norm_n = list()
_y_n =list()
h = list()

for i in range(1, channel_num+1):
    path = f'./SIMO_data/2dBTrial/ISAC_QPSK_OFDM_{i}.mat'
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
    _attenuation.append(
        data['ISAC_data']['channel'][0][0]['attenuation'][0][0][0]
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
    _y_n.append(
        data['ISAC_data']['y_n'][0][0]
    )
    _y_norm_n.append(
        data['ISAC_data']['y_norm_n'][0][0]
    )
    h.append(
        data['ISAC_data']['h'][0][0]
    )

_channel['time_delay'] = _tau
_channel['doppler_shift'] = _fdoppler
_channel['_Txsteering'] = _Txsteering
_channel['_Rxsteering'] = _Rxsteering

x_simo, _y, y_simo, h = np.array(_x), np.array(_y), np.array(_y_norm), np.array(h)
y_simo_n = np.array(_y_n)
_x = np.array(_x)

y_simo_isac = np.zeros(shape = (carrier_num, channel_num, symbol_num, Nr, 2))
y_simo_isac_org =  np.zeros(shape = (carrier_num, channel_num, symbol_num, Nr, 2))
label_simo_isac = np.zeros(shape = (carrier_num, channel_num, symbol_num))
for n in range (carrier_num):
    for i in range (channel_num):
        for j in range (symbol_num):
            label_simo_isac[n, i, j] = mapping_qpsk (
                x_simo[i, j, n]
            )
            for k in range (Nr):
                y_simo_isac[n, i, j, k] = np.array(
                    [
                        np.real(y_simo_n[i, j, n, k]),
                        np.imag(y_simo_n[i, j, n, k]),
                    ]
                )

                y_simo_isac_org[n, i, j, k] = np.array([
                    np.real(_y[i, j, n, k]),
                    np.imag(_y[i, j, n, k])
                ])


X_noisy = torch.tensor(
    y_simo_isac, 
    dtype = torch.float32
).permute(1, 0, 2, 3, 4)

X = torch.tensor(
    y_simo_isac_org, 
    dtype = torch.float32
).permute(1, 0, 2, 3, 4)

_idx = int(0.8 * channel_num)
X_noisy_train = X_noisy[0: _idx]
X_train = X[0: _idx]
from torch.utils.data import DataLoader, TensorDataset

train = DataLoader(
    dataset=TensorDataset(X_noisy_train, X_train),
    batch_size=50,
    shuffle = True,
    drop_last = True
)

X_noisy_test = X_noisy[_idx:]
X_test = X[_idx:]

test = DataLoader(
    dataset=TensorDataset(X_noisy_test, X_test),
    batch_size = 10,
    shuffle = True,
    drop_last=True
)
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Initialize model, criterion, and optimizer
input_channels = 2  # Real and imaginary parts
output_channels = 2
print (device)
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

import hiddenlayer as hl

historyl = hl.History()
cavasl = hl.Canvas()
print (
    f'original MMSE loss on test data: {torch.sqrt(criterion (X_noisy_test, X_test)).item():.4f}'
)
# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    train_loss_epoch = 0
    test_loss_epoch = 0
    train_num = 0
    test_num = 0

    for step, (_X, _Y) in enumerate(train):
        outputs = model(_X)
        loss = criterion(outputs, _Y)
        train_num += 1

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_mmse = torch.sqrt(loss)
        train_loss_epoch += loss_mmse

    for step, (_X_test, _Y_test) in enumerate(test):
        _X_test = _X_test
        _Y_test = _Y_test
        out = model(_X_test)

        _loss = criterion(out, _Y_test)

        test_loss_epoch += torch.sqrt(_loss)
        test_num += 1


    if (epoch+1) % 1 == 0:
        print ('- - - - - - - - - - - - - - - -')
        print(f'Epoch [{epoch+1}/{num_epochs}],Train Loss: {train_loss_epoch / (train_num):.4f}')
        print(f'Epoch [{epoch+1}/{num_epochs}],Test Loss: {test_loss_epoch / (test_num):.4f}')
        print ('\n')

    historyl.log(
        epoch,
        train = train_loss_epoch / (train_num),
        test = test_loss_epoch / (test_num),
        LS_test = torch.sqrt(criterion (X_noisy_test, X_test)).item()
    )

    cavasl.draw_plot(
        [
            historyl['train'],
            historyl['test'],
            historyl['LS_test']
        ],
        _title = 'MMSE of CSI for training and testing sample',
        ylabel = 'MMSE'
    )

    cavasl.save('./results/training_CE.png', dpi = 300)
torch.save(model.state_dict(), './model/simo_isac_denoising.pt')
