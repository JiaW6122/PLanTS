import torch
import torch.nn as nn



class DynamicCondPredHead(nn.Module):

    def __init__(self, in_features, out_features, hidden_features, dropout=0.1):
        super(DynamicCondPredHead, self).__init__()
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
