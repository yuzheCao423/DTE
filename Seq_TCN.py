import torch
import torch.nn as nn
import math
from .feature_utils import get_lags, add_lag_features, add_fourier_features
from .positional_encodings import Sin_Net, Cos_Net, Add_Emb, Simple_Net
from .combined_rnn import (
    TCN_RNN_Layer, TCN_GRU_Layer, TCN_Linear_Layer, RNN_GRU_Layer, TCN_LSTM_Layer
)

class Seq_TCN(nn.Module):
    def __init__(
        self,
        prediction_length,
        input_size=1,
        hidden_size=48,
        output_size=1,
        context_length=300,
        tangle_channels=30,
        cell_type="rnn",
        tcn_channels=32,
        tcn_layers=6,
        rnn_hidden_size=64,
        rnn_layers=4,
        freq="1H",
        embed_size=16,
        lee_size=0,
        learned_embedding_dim=0,
        fourier_dim=0,  # must be even
        useLags=False
    ):
        super(Seq_TCN, self).__init__()
        self.tangle_channels = tangle_channels
        self.useLags = useLags

        # 计算最终输入维度
        lag_len = len(get_lags(context_length)) if self.useLags else 0
        self.Z_dim = (input_size + 2 * tangle_channels + embed_size +
                      lee_size + learned_embedding_dim + fourier_dim + lag_len)

        if self.tangle_channels > 0:
            self.sin_net = Sin_Net(channel=tangle_channels, context_length=context_length)
            self.cos_net = Cos_Net(channel=tangle_channels, context_length=context_length)

        # 选择时序处理单元
        if cell_type == "rnn":
            self.tcn_rnn = TCN_RNN_Layer(
                tcn_input_size=self.Z_dim,
                tcn_channels=tcn_channels,
                tcn_layers=tcn_layers,
                rnn_hidden_size=rnn_hidden_size,
                rnn_layers=rnn_layers,
                linear_output_size=hidden_size,
            )
        elif cell_type == "gru":
            self.tcn_rnn = TCN_GRU_Layer(
                tcn_input_size=self.Z_dim,
                tcn_channels=tcn_channels,
                tcn_layers=tcn_layers,
                rnn_hidden_size=rnn_hidden_size,
                rnn_layers=rnn_layers,
                linear_output_size=hidden_size,
            )
        elif cell_type == "Linear":
            self.tcn_rnn = TCN_Linear_Layer(
                tcn_input_size=self.Z_dim,
                tcn_channels=tcn_channels,
                tcn_layers=tcn_layers,
                rnn_hidden_size=rnn_hidden_size,
                rnn_layers=rnn_layers,
                linear_output_size=hidden_size,
                context_length=context_length,
            )
        elif cell_type == "rnn_gru":
            self.tcn_rnn = RNN_GRU_Layer(
                input_size=self.Z_dim,
                rnn_channels=tcn_channels,
                tcn_layers=tcn_layers,
                rnn_hidden_size=rnn_hidden_size,
                rnn_layers=rnn_layers,
                linear_output_size=hidden_size,
            )
        elif cell_type == "lstm":
            self.tcn_rnn = TCN_LSTM_Layer(
                tcn_input_size=self.Z_dim,
                tcn_channels=tcn_channels,
                tcn_layers=tcn_layers,
                rnn_hidden_size=rnn_hidden_size,
                rnn_layers=rnn_layers,
                linear_output_size=hidden_size,
            )

        if embed_size > 0:
            self.add_emb = Add_Emb(context_length=context_length, add_size=embed_size)
        self.embed_size = embed_size

        if lee_size > 0:
            self.lee = Simple_Net(context_length=context_length, output_size=lee_size)
        self.lee_size = lee_size

        self.prediction_length = prediction_length
        self.input_size = input_size
        self.output_size = output_size
        self.context_length = context_length
        self.nn = nn.Linear(hidden_size, input_size)
        self.learned_embedding_dim = learned_embedding_dim
        self.fourier_dim = fourier_dim
        if fourier_dim > 0 and fourier_dim % 2 != 0:
            raise ValueError("fourier_dim must be even")

        if self.learned_embedding_dim > 0:
            self.learned_embedding = nn.Embedding(context_length+prediction_length,
                                                   learned_embedding_dim)

    def freq_to_parameter(self, freq):
        if freq == "1H":
            return 1
        elif freq == "1D":
            return 24
        elif freq == "1W":
            return 24 * 7
        elif freq == "1M":
            return 24 * 30
        elif freq == "1Y":
            return 24 * 365
        else:
            raise ValueError("freq must be in ['1H', '1D', '1W', '1M', '1Y']")

    def forward(self, x):
        device = x.device
        x = x.unsqueeze(dim=-1)  # [batch, seq, 1]
        all_position = torch.arange(self.context_length + self.prediction_length).to(device)

        output_seq = []
        for i in range(self.prediction_length):
            position = all_position[i: i + self.context_length]

            if self.embed_size > 0:
                x_plus_emb = self.add_emb(x)
            else:
                x_plus_emb = x

            if self.lee_size > 0:
                x_plus_lee = self.lee(x)
                x_plus_lee = torch.cat([x_plus_emb, x_plus_lee], dim=-1)
            else:
                x_plus_lee = x_plus_emb

            if self.tangle_channels != 0:
                sin_position = self.sin_net(x, position)
                cos_position = self.cos_net(x, position)
                x_plus_position = torch.cat([x_plus_lee, sin_position, cos_position], dim=-1)
            else:
                x_plus_position = x_plus_lee

            if self.learned_embedding_dim > 0:
                batch_position = position.unsqueeze(0).repeat(x.shape[0], 1)
                learned_emb = self.learned_embedding(batch_position)
                x_plus_embedding = torch.cat([x_plus_position, learned_emb], dim=-1)
            else:
                x_plus_embedding = x_plus_position

            if self.fourier_dim > 0:
                x_plus_fourier = add_fourier_features(x_plus_embedding,
                                                       num_freqs=self.fourier_dim // 2)
            else:
                x_plus_fourier = x_plus_embedding

            if self.useLags:
                lags = get_lags(self.context_length)
                x_plus_lag = add_lag_features(x_plus_fourier, lags)
            else:
                x_plus_lag = x_plus_fourier

            x_all = x_plus_lag
            y = self.tcn_rnn(x_all)          # [batch, hidden_size]
            y = y.unsqueeze(1)                # [batch, 1, hidden_size]
            output_seq.append(y)

            x_next = self.nn(y)                # [batch, 1, input_size]
            x = torch.cat([x[:, 1:, :], x_next], dim=1)

        return torch.cat(output_seq, dim=1)   # [batch, prediction_length, hidden_size]