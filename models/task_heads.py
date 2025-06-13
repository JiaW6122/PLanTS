import torch
import torch.nn as nn
from models.dilated_conv import DilatedConvDecoder

class DynamicCondPredHead(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_features[0]),
            self.relu,
            nn.Linear(hidden_features[0], out_features),
            self.relu,
            self.dropout,

        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.fc(x)

class DynamicConstructHead(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_features[0]),
            # self.relu,
            nn.Linear(hidden_features[0], out_features),
            self.relu,
            self.dropout,

        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.fc(x)


class DynamicPredHead(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, dropout=0.1):
        super(DynamicPredHead, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_features[0]),
            self.relu,
            nn.Linear(hidden_features[0], out_features),
            self.relu,
            self.dropout,

        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.fc(x)

class DynamicConstructPredHead(nn.Module):
    def __init__(self, d_static, d_time, hidden_dims, out_channels, kernel_size=3):
        super().__init__()
        input_dim = d_static + d_time
        self.decoder = DilatedConvDecoder(
            in_channels=input_dim,
            channels=hidden_dims + [out_channels],
            kernel_size=kernel_size
        )

    def forward(self, x_static, tmp_emb):
        x = torch.cat([x_static, tmp_emb], dim=-1)  # (B, scale, d_static + d_time)
        x = x.transpose(1, 2)                       # (B, D, scale)
        x = self.decoder(x)
        return x.transpose(1, 2)                    # (B, scale, out_channels)
