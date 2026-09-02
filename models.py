import math
import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # take last time-step
        last = out[:, -1, :]
        return self.fc(last)


class CNN1DForecaster(nn.Module):
    def __init__(self, input_channels=1, channels=[16, 32], kernel_size=3, seq_len=50):
        super().__init__()
        layers = []
        in_ch = input_channels
        for ch in channels:
            layers.append(nn.Conv1d(in_ch, ch, kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(ch))
            in_ch = ch
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch, 1)

    def forward(self, x):
        # x: (batch, seq_len, channels) -> conv expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        out = self.net(x)
        out = self.pool(out).squeeze(-1)
        return self.fc(out)


class StateSpaceForecaster(nn.Module):
    """Simple learnable linear state-space model:
    x_{t+1} = A x_t + B u_t
    y_t = C x_t + D u_t
    where u_t is input (observations) and y_t is scalar target.
    """
    def __init__(self, state_dim=32, input_dim=1):
        super().__init__()
        self.state_dim = state_dim
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * 0.1)
        self.C = nn.Parameter(torch.randn(1, state_dim) * 0.1)
        self.D = nn.Parameter(torch.randn(1, input_dim) * 0.1)

    def forward(self, u):
        # u: (batch, seq_len, input_dim)
        batch, seq_len, input_dim = u.shape
        x = u.new_zeros(batch, self.state_dim)
        for t in range(seq_len):
            ut = u[:, t:t+1, :].reshape(batch, input_dim)
            x = torch.matmul(x, self.A.t()) + torch.matmul(ut, self.B.t())
        y = torch.matmul(x, self.C.t()) + torch.matmul(ut, self.D.t())
        return y.squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerForecaster(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        last = h[:, -1, :]
        return self.fc(last)
