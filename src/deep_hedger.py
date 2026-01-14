import torch
import torch.nn as nn

class DeepHedgingModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=1):
        super(DeepHedgingModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        # CORRECTION 1 : On utilise LayerNorm, pas BatchNorm
        # LayerNorm respecte la logique temporelle du LSTM
        self.layer_norm = nn.LayerNorm(input_dim + 1)
        
        self.lstm_cell = nn.LSTMCell(input_size=input_dim + 1, hidden_size=hidden_dim)
        
        self.decision_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
  
            nn.Sigmoid() # On revient à Sigmoid (plus stable pour le gradient que Hardtanh ici)
        )

    def forward(self, x, initial_position=None):
        batch_size, seq_len, _ = x.size()
        
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        if initial_position is None:
            prev_delta = torch.zeros(batch_size, 1, device=x.device)
        else:
            prev_delta = initial_position

        deltas = []
        
        for t in range(seq_len):
            market_features = x[:, t, :]
            
            combined_input = torch.cat([market_features, prev_delta], dim=1)
            
            # CORRECTION 1 (Suite) : Normalisation propre
            combined_input = self.layer_norm(combined_input)
            
            h_t, c_t = self.lstm_cell(combined_input, (h_t, c_t))
            current_delta = self.decision_layer(h_t)
            
            prev_delta = current_delta
            deltas.append(current_delta)
            
        return torch.stack(deltas, dim=1)