import torch
import torch.nn as nn

class Sin_Net(nn.Module):
    def __init__(self, channel=1, context_length=300):
        super(Sin_Net, self).__init__()
        self.w = nn.Parameter(torch.rand([1, channel]) * 2 * 3.14 / 24)
        self.b = nn.Parameter(torch.rand([1, channel]) * 2 * 3.14)
        self.A = nn.Parameter(torch.ones([1, channel]) * 0.5)
        self.B = nn.Parameter(torch.rand([1, channel]) * 0.5)
        self.channel = channel

    def forward(self, x, position):
        if self.channel == 0:
            return x
        device = x.device
        out = []
        batch_size = x.shape[0]
        seq_length = x.shape[1]

        for i in range(self.channel):
            w = self.w[:, i]
            b = self.b[:, i]
            A = self.A[:, i]
            B = self.B[:, i]
            wx_b = position * w + b
            Asin_wx_b = A * torch.sin(wx_b) + B
            Asin_wx_b = Asin_wx_b.reshape(1, -1)
            batchified = Asin_wx_b.repeat(batch_size, 1)
            out.append(batchified)
        out = torch.stack(out, dim=-1)
        return out

class Cos_Net(nn.Module):
    def __init__(self, channel=1, context_length=300):
        super(Cos_Net, self).__init__()
        self.w = nn.Parameter(torch.rand([1, channel]) * 2 * 3.14 / 24)
        self.b = nn.Parameter(torch.rand([1, channel]) * 2 * 3.14)
        self.A = nn.Parameter(torch.ones([1, channel]) * 0.5)
        self.B = nn.Parameter(torch.rand([1, channel]) * 0.5)
        self.channel = channel

    def forward(self, x, position):
        if self.channel == 0:
            return x
        device = x.device
        out = []
        batch_size = x.shape[0]
        seq_length = x.shape[1]

        for i in range(self.channel):
            w = self.w[:, i]
            b = self.b[:, i]
            A = self.A[:, i]
            B = self.B[:, i]
            wx_b = position * w + b
            Acos_wx_b = A * torch.cos(wx_b)
            Acos_wx_b = Acos_wx_b.reshape(1, -1) + B
            batchified = Acos_wx_b.repeat(batch_size, 1)
            out.append(batchified)
        out = torch.stack(out, dim=-1)
        return out

class Add_Emb(nn.Module):
    def __init__(self, context_length=300, add_size=64):
        super(Add_Emb, self).__init__()
        self.emb = nn.Parameter(torch.rand(context_length, add_size))

    def forward(self, x):
        batchified_emb = self.emb.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)
        x = torch.cat([x, batchified_emb], dim=-1)
        return x

class Simple_Net(nn.Module):
    def __init__(self, context_length=300, output_size=3):
        super(Simple_Net, self).__init__()
        self.linear = nn.Linear(1, output_size)

    def forward(self, x):
        simple_emb = self.linear(x)
        return simple_emb

class Freq_Emb(nn.Module):
    def __init__(self, context_length=300, output_size=3):
        super(Freq_Emb, self).__init__()
        self.linear = nn.Linear(context_length, context_length * output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        X = torch.fft.fft(x, dim=1)
        X = torch.abs(X)
        X = X.reshape(batch_size, -1)
        fft_emb = self.linear(X)
        fft_emb = fft_emb.reshape(batch_size, seq_length, -1)
        return fft_emb

class Linear_Emb(nn.Module):
    def __init__(self, context_length=300, output_size=3):
        super(Linear_Emb, self).__init__()
        self.linear = nn.Linear(context_length, context_length * output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        x = x.reshape(batch_size, -1)
        linear_emb = self.linear(x)
        linear_emb = linear_emb.reshape(batch_size, seq_length, -1)
        return linear_emb
