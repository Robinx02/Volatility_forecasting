import torch
import torch.nn as nn


class LSTMVolatilityModel(nn.Module):
    """
    2-layer LSTM for realized volatility forecasting.
    Input  : (batch, seq_len, n_features)
    Output : (batch, 1)  — point forecast of rv at t+1
    """

    def __init__(self, n_features: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size   = n_features,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            dropout      = dropout,          
            batch_first  = True,
        )

        self.dropout = nn.Dropout(dropout)   
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # the variable x: (batch, seq_len, n_features)
        
        out, _ = self.lstm(x)                
        last    = out[:, -1, :]              
        last    = self.dropout(last)
        return self.fc(last).squeeze(-1)     