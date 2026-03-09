import torch
import torch.nn as nn
import torch.nn.functional as F

class NextFrameLSTM(nn.Module):
    #construct the model
    def __init__(self, n_mfcc: int, hidden_size: int =256, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=n_mfcc,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout = dropout if num_layers > 1 else 0.0,
            bidirectional=False,
            batch_first=True,
        )
        #if batch first true -> (batch, Time, Features)
        #project hidden to mfcc
        self.to_mfcc = nn.Linear(hidden_size, n_mfcc)

    def forward(self,x,h0=None):
        out, (h_T,c_T) = self.lstm(x,h0)
        y_pred =self.to_mfcc(out)
        return y_pred

