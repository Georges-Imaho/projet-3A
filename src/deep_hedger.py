import torch
import torch.nn as nn

class DeepHedgingModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=1):
        super(DeepHedgingModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        # On ajoute +1 à l'input_dim pour inclure le "Previous Delta"
        self.lstm_cell = nn.LSTMCell(input_size=input_dim + 1, hidden_size=hidden_dim)
        
        self.decision_layer = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid() 
        )

    def forward(self, x, initial_position=None):
        """
        x shape: [Batch, Time, Features]
        """
        batch_size, seq_len, _ = x.size()
        
        # Initialisation des états cachés (h_t, c_t)
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        # Position initiale (0 actions au début)
        if initial_position is None:
            prev_delta = torch.zeros(batch_size, 1, device=x.device)
        else:
            prev_delta = initial_position

        deltas = []
        
        # --- BOUCLE TEMPORELLE (On avance jour après jour) ---
        for t in range(seq_len):
            # 1. On récupère les features du marché du jour t
            market_features = x[:, t, :] # [Batch, Features]
            
            # 2. On concatène avec la position de la veille (L'IA sait ce qu'elle a en poche)
            combined_input = torch.cat([market_features, prev_delta], dim=1) 
            
            # 3. Update de la mémoire LSTM
            h_t, c_t = self.lstm_cell(combined_input, (h_t, c_t))
            
            # 4. Décision d'action
            current_delta = self.decision_layer(h_t)
            
            # 5. On sauvegarde pour la boucle suivante et pour la sortie
            prev_delta = current_delta
            deltas.append(current_delta)
            
        # On recolle tout sous forme [Batch, Time, 1]
        return torch.stack(deltas, dim=1)

# --- Test rapide pour vérifier que les dimensions collent ---
if __name__ == "__main__":
    # Simulation d'un batch de données
    batch_size = 64
    seq_len = 30     # 30 jours
    n_features = 3   # (Log-Price, Time-to-Maturity, Volatility)
    
    # Création du modèle
    model = DeepHedgingModel(input_dim=n_features)
    
    # Création d'une entrée aléatoire (Dummy data)
    dummy_input = torch.randn(batch_size, seq_len, n_features)
    
    # Passage dans le modèle
    output = model(dummy_input)
    
    print(f"Input shape  : {dummy_input.shape}") # [64, 30, 3]
    print(f"Output shape : {output.shape}")      # [64, 30, 1]
    print("✅ Le modèle fonctionne techniquement (les dimensions sont correctes).")