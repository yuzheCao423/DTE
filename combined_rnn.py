import torch
import torch.nn as nn
import math
from .tcn_base import TCN

class TCN_RNN(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super(TCN_RNN, self).__init__()
        self.tcn = TCN(num_inputs=input_size, channels=[32] * 6)
        self.rnn = nn.RNN(input_size=32, hidden_size=64, num_layers=4, batch_first=True)
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

class TCN_RNN_Layer(nn.Module):
    def __init__(
        self,
        tcn_input_size=1,
        tcn_output_size=1,
        tcn_channels=32,
        tcn_layers=6,
        rnn_hidden_size=64,
        rnn_layers=4,
        linear_output_size=1,
    ):
        super(TCN_RNN_Layer, self).__init__()
        self.tcn = TCN(num_inputs=tcn_input_size, channels=[tcn_channels] * tcn_layers)
        self.rnn = nn.RNN(
            input_size=tcn_channels,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(rnn_hidden_size, linear_output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

class TCN_LSTM_Layer(nn.Module):
    def __init__(
        self,
        tcn_input_size=1,
        tcn_output_size=1,
        tcn_channels=32,
        tcn_layers=6,
        rnn_hidden_size=64,
        rnn_layers=4,
        linear_output_size=1,
    ):
        super(TCN_LSTM_Layer, self).__init__()
        self.tcn = TCN(num_inputs=tcn_input_size, channels=[tcn_channels] * tcn_layers)
        self.lstm = nn.LSTM(
            input_size=tcn_channels,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(rnn_hidden_size, linear_output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

class RNN_GRU_Layer(nn.Module):
    def __init__(
        self,
        input_size=1,
        rnn_channels=32,
        tcn_layers=6,
        rnn_hidden_size=64,
        rnn_layers=4,
        linear_output_size=1,
    ):
        super(RNN_GRU_Layer, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=rnn_channels,
            num_layers=tcn_layers,
            batch_first=True,
        )
        self.gru = nn.GRU(
            input_size=rnn_channels,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(rnn_hidden_size, linear_output_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

class TCN_GRU_Layer(nn.Module):
    def __init__(
        self,
        tcn_input_size=1,
        tcn_output_size=1,
        tcn_channels=32,
        tcn_layers=6,
        rnn_hidden_size=64,
        rnn_layers=4,
        linear_output_size=1,
    ):
        super(TCN_GRU_Layer, self).__init__()
        self.tcn = TCN(num_inputs=tcn_input_size, channels=[tcn_channels] * tcn_layers)
        self.gru = nn.GRU(
            input_size=tcn_channels,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(rnn_hidden_size, linear_output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

class TCN_Linear_Layer(nn.Module):
    def __init__(
        self,
        tcn_input_size=1,
        tcn_output_size=1,
        tcn_channels=32,
        tcn_layers=6,
        rnn_hidden_size=64,
        rnn_layers=4,
        linear_output_size=1,
        context_length=300,
    ):
        super(TCN_Linear_Layer, self).__init__()
        self.tcn = TCN(num_inputs=tcn_input_size, channels=[tcn_channels] * tcn_layers)
        self.mlp = nn.Sequential(
            nn.Linear(tcn_channels * context_length, rnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(rnn_hidden_size, rnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fc = nn.Linear(rnn_hidden_size, linear_output_size)

    def forward(self, x):
        B = x.shape[0]
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(B, -1)
        x = self.mlp(x)
        x = self.fc(x)
        return x

class TCN_GRU(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super(TCN_GRU, self).__init__()
        self.tcn = TCN(num_inputs=input_size, channels=[32] * 3)
        self.gru = nn.GRU(input_size=32, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, int(math.sqrt(output_size)))
        self.fc = nn.Linear(int(math.sqrt(output_size)), output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

class TCN_LSTM(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super(TCN_LSTM, self).__init__()
        self.tcn = TCN(num_inputs=input_size, channels=[32, 32, 32])
        self.lstm = nn.LSTM(
            input_size=32, hidden_size=64, num_layers=2, batch_first=True
        )
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x