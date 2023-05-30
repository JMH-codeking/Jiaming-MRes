import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class LSTM_net(nn.Module):

    def __init__(self): 
        # input the dimensio  n for input and output antenna number
        super(LSTM_net, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size = 2,
            hidden_size = 8, # let h initial be 
            num_layers = 6,
            bidirectional = True,
            batch_first = True,
            dropout = 0.2
        )
        self.lstm2 = nn.LSTM(
            input_size = 8,
            hidden_size = 8,
            num_layers = 3,
            bidirectional = True,
            batch_first = True,
            dropout = 0.5
        )
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=2)
        self.conv2 = nn.Conv1d(8, 16, 2)
        self.deconv1 = nn.ConvTranspose1d(16, 8, kernel_size=2)
        self.deconv2 = nn.ConvTranspose1d(8, 4, kernel_size=2)
        self.fc1 = nn.Linear(in_features=16, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=4)
    def forward(self, x):
        batch_size, seq_len, input_len = x.shape
        h0 = Variable(torch.randn((2 * self.lstm1.num_layers, batch_size, self.lstm1.hidden_size)))
        c0 = Variable(torch.randn((2 * self.lstm1.num_layers, batch_size, self.lstm1.hidden_size)))
        x, (h1, c1) = self.lstm1(x, (h0, c0))
        x = torch.sigmoid(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x= torch.tanh(x)
        # x, (h3, c3) = self.lstm2(x, (h1, c1))
        # x = self.fc1(x)
        # x = x.reshape(10, 64, 4, 2)

        # x = self.dnn(x)
        # h0 = Variable(torch.randn((6, 64, 4)))
        # c0 = Variable(torch.randn((6, 64, 4)))
        # x, (final_hidden_state, final_cell_state) = self.lstmnet(x, (h0, c0))
        # x = self.fc2(x)
        return x

class Channel_ana(nn.Module):
    '''a model for channel analysis in blind signal deconvolution
    
    take a random Mr * Mt input _k for Net_K

    '''

    def __init__(self):
        super(Channel_ana, self).__init__()
        self.Encoder = nn.Sequential(
            # encoder
            nn.Conv2d(
                in_channels=1,
                out_channels = 16,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.Tanh(),
            # nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.Tanh(),
            # nn.BatchNorm2d(32),
        )

        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 1, 1),
            nn.Tanh(),
            # nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, h):
        h = self.Encoder(h)
        h = self.Decoder(h)

        return h

class Signal_ana(nn.Module):
    '''a model for signal analysis in blind signal deconvolution

    take a random Mt * Q input _x for Net_X

    combining _x and _k:
        y_hat = _x * _k
        minimising (y - y_hat) -> loss = mse(y, y_hat)
    '''
    def __init__(self):
        super(Signal_ana, self).__init__()
        self.Encoder = nn.Sequential(
            # encoder
            nn.Conv2d(
                in_channels=1,
                out_channels = 16,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.Tanh(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.Tanh(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.Tanh(),
            nn.BatchNorm2d(64),
        )

        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 1, 1),
            nn.Tanh(),
            nn.ConvTranspose2d(32, 16, 3, 1, 1),
            nn.Tanh(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)

        return x

class Classifier(nn.Module):
    '''a model for detecting time delays
    
    '''

    def __init__(self):
        super(Classifier, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels = 16,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32)     
        )

        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.ConvTranspose2d(16, 1, 3, 1, 1),
            nn.Tanh(),
            
        )
        self.fc = nn.Linear(4*32, 2)

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        x = x.view(-1, 4 * 32)
        x = self.fc(x)
        return x


class Encoder_Decoder(nn.Module):
    '''a model for signal reconstruction
    
    '''

    def __init__(self):

        super(Encoder_Decoder, self).__init__()
        
        self.Encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels = 16,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.Tanh(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.Tanh(),
            nn.BatchNorm2d(32)     
        )

        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 1, 1),
            nn.Tanh(),
            nn.BatchNorm2d(16),

            nn.ConvTranspose2d(16, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x


# class config():
#     def __init__(
#         self,
#         input_channels,
#         output_channels,
#         embed,
#         dropout,
#     ) -> None:
        
#         self.input_channels = input_channels
#         self.output_channels = output_channels
#         self.embed = embed
#         self.dropout = dropout    

# class neural_network(nn.Module):
#     def __init__(self, config) -> None:
#         super(neural_network, self).__init__()
#         if config.embedding_pretrained is not None:
#             self.embedding = nn.Embedding.from_pretrained(
#                 config.embedding_pretrained, freeze=False
#             )
#         else:
#             self.embedding = nn.Embedding(
#                 config.n_vocab, config.embed, padding_idx = config.n_vocab - 1
#             )

#         self.lstm = nn.LSTM(
#             config.embed, # input size for each cell
#             config.hidden_size, config.num_layers, 
#             bidirectional = True, batch_first = True, dropout = config.dropout
#         )
#         self.fc = nn.Linear(config.hidden_size*2, config.num_classes)

#     def forward(self, x):
#         x, _ = x
#         x = self.embedding(x)
#         x, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])
#         return out
    