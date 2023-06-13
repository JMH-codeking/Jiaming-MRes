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
    
def cross_entropy(y_true,y_pred):
    C=0
    # one-hot encoding
    for col in range(y_true.shape[-1]):
        y_pred[col] = y_pred[col] if y_pred[col] < 1 else 0.99999
        y_pred[col] = y_pred[col] if y_pred[col] > 0 else 0.00001
        C+=y_true[col]*torch.log(y_pred[col])+(1-y_true[col])*torch.log(1-y_pred[col])
    return -C

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
    path = f'./SIMO_data/0dB/ISAC_QPSK_OFDM_{i}.mat'
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
y_simo_n = np.array(_y_norm_n)



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
                    np.real(y_simo[i, j, n, k]),
                    np.imag(y_simo[i, j, n, k])
                ])


X_noisy = torch.tensor(
    y_simo_isac, 
    dtype = torch.float32
).permute(1, 4, 0, 2, 3)

X = torch.tensor(
    y_simo_isac_org, 
    dtype = torch.float32
).permute(1, 4, 0, 2, 3)

X_noisy_train = X_noisy[0:900]
X_train = X[0:900]
from torch.utils.data import DataLoader, TensorDataset

train = DataLoader(
    dataset=TensorDataset(X_noisy_train, X_train),
    batch_size=50,
    shuffle = True,
    drop_last = True
)

X_noisy_test = X_noisy[950:].contiguous().view(-1, 2, 16, 50).to(device)
X_test = X[950:].contiguous().view(-1, 2, 16, 50).to(device)
import torch.nn as nn
import torch.optim as optim
class UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc_conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Decoder
        self.dec_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv1 = nn.Conv2d(192, 64, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.dec_conv3 = nn.Conv2d(32, output_channels, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoder
        enc1 = self.relu(self.enc_conv1(x))
        enc2 = self.relu(self.enc_conv2(enc1))
        enc_pool = self.enc_pool(enc2)

        # Bottleneck
        bottleneck = self.relu(self.bottleneck_conv(enc_pool))

        # Decoder
        dec_upsample = self.dec_upsample(bottleneck)
        dec1 = self.relu(self.dec_conv1(torch.cat([dec_upsample, enc2], dim=1)))
        dec2 = self.relu(self.dec_conv2(dec1))
        dec3 = self.dec_conv3(dec2)
        
        return dec3

# Initialize model, criterion, and optimizer
input_channels = 2  # Real and imaginary parts
output_channels = 2
model = UNet(input_channels, output_channels).to(device=device)
if torch.cuda.is_available():
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

import hiddenlayer as hl

historyl = hl.History()
cavasl = hl.Canvas()
print (f'original MSE loss on test data: {criterion (X_noisy_test, X_test)}')
# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    train_loss_epoch = 0
    for step, (_X, _Y) in enumerate(train):
        _X = _X.contiguous().view(-1, 2, 16, train.batch_size).to(device)
        _Y = _Y.contiguous().view(-1, 2, 16, train.batch_size).to(device)
    # Forward pass
        outputs = model(_X)
        loss = criterion(outputs, _Y)
        train_loss_epoch += loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss_epoch / (step+1):.4f}')
    
    historyl.log(
        epoch,
        loss = loss
    )

    cavasl.draw_plot(
        historyl['loss']
    )
out = model(X_noisy_test)
print (criterion(out, X_test).item())
print('Finished Training')
