from __future__ import annotations
import torch
import torch.nn as nn

class TemporalEncoder(nn.Module):
    """
    CNN + BiLSTM over time for each sensor.
    Input:  X_in (N, T, D)
    Output: z_t (N, T, latent)
    """
    def __init__(self, in_dim: int, cnn_channels: int, cnn_kernel: int, lstm_hidden: int, latent: int, lstm_layers: int = 1):
        super().__init__()
        padding = cnn_kernel // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=cnn_channels, kernel_size=cnn_kernel, padding=padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=cnn_kernel, padding=padding),
            nn.ReLU(),
        )
        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(2 * lstm_hidden, latent)

    def forward(self, X_in: torch.Tensor) -> torch.Tensor:
        # X_in: (N,T,D)
        N, T, D = X_in.shape
        x = X_in.transpose(1, 2)        # (N,D,T)
        x = self.cnn(x)                  # (N,cnn,T)
        x = x.transpose(1, 2)            # (N,T,cnn)
        y, _ = self.bilstm(x)            # (N,T,2*lstm_hidden)
        z = self.proj(y)                 # (N,T,latent)
        return z
