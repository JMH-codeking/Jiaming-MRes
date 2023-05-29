
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
    valid_data,
    model, 
    num_epoch = 1000, 
    _loss = 'mse',
    save_path = './model/model_time_delay.pt'
):
    if _loss == "mse":
        loss_function = nn.MSELoss()
    elif _loss == "cross_entropy":
        loss_function = nn.CrossEntropyLoss()
    else:
        assert 'Wrong Name'

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr = 1e-3, 
        weight_decay = 1e-5
    )
    
    historyl = hl.History()
    canvasl = hl.Canvas()

    train_num = 0

    for epoch in range (num_epoch):
        print (f'-----Epoch {epoch + 1}-----')
        train_loss_epoch = 0

        '''train

        training process -> normal network forward and backward optimisation
        '''
        model.train()
        accuracy = 0
        total = 0
        for step, (train_x, labels) in enumerate(train_data):
            output = model(train_x)
            loss = loss_function(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item() * train_x.size(0)
            train_num = train_num + train_x.size(0)
           
        train_loss = train_loss_epoch / train_num

        '''valid
        
        valiation process -> no network forward and backward process
        '''
        model.eval()
        accuracy = 0
        total = 0
        with torch.no_grad():
            for step, (valid_x, valid_y) in enumerate(valid_data):
                output = model(valid_x)
                _, predicted = torch.max(output.data, 1)
                total+= valid_y.size(0)
                accuracy += (predicted == valid_y).sum().item()

        accuracy = (100 * accuracy / total)
        print (f'accuracy is: {accuracy}')
        historyl.log(
            epoch, 
            train_loss = train_loss,
            valid_accuracy = accuracy,
        )
        with canvasl:
            canvasl.draw_plot(
                [
                    historyl['train_loss']
                ]
            )
            canvasl.draw_plot(
                [
                    historyl['valid_accuracy']
                ]
            )
        torch.save(model.state_dict(), save_path)


def main():
    current_path = str(pathlib.Path(__file__).parent)
    transmitted_data = f'{current_path}/data/transmitted_data.mat'
    received_data = f'{current_path}/data/received_data.mat'
    batch_size = 25

    constellation_data, mapped_constellation, x_transmitted, channel_matrix, \
      received_data_no_noise, received_data_with_noise \
            = matlab_data_extraction(transmitted_data, received_data)
    
    received_data_real = complex2real(received_data_with_noise)

    received_data_td, label = time_delay_addition_label(
        received_data_real,
        10,
        noise_sigma = 0.1,
    ) # received_data_td is `x`, and `y` is the label

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        received_data_td, label, test_size = 0.25, 
    )

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size = 0.25, 
    )

    # change shape into CNN-style -> num_data, channel_num, l, w
    h_x = x_train.shape[0]
    l_x = x_train.shape[1]
    w_x = x_train.shape[2]
    x_train = torch.tensor(x_train, dtype=torch.float32).reshape(
        h_x, 1, l_x, w_x
    )
    y_train = torch.LongTensor(y_train)

    h_x_v = x_valid.shape[0]
    l_x_v = x_valid.shape[1]
    w_x_v = x_valid.shape[2]
    x_valid = torch.tensor(x_valid, dtype=torch.float32).reshape(
        h_x_v, 1, l_x_v, w_x_v
    )
    y_valid = torch.LongTensor(y_valid)

    train_loader = DataLoader(
        dataset = TensorDataset(x_train, y_train),
        batch_size = batch_size,
        shuffle = True,
    )
    valid_loader = DataLoader(
        dataset = TensorDataset(x_valid, y_valid),
        batch_size = batch_size,
        shuffle = True,
    )

    '''model training
    
    '''


    model = Classifier()
    from torchsummary import summary
    # summary(model, input_size=(1, 4, 32))

    train_valid(
        train_loader,
        valid_loader,
        model,
        num_epoch=10,
        _loss = 'cross_entropy'
    )



if __name__ == "__main__":
    
    main()