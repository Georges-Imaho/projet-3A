import torch
import torch.nn as nn

class DeepHedgingModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=1):
        super(DeepHedgingModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.bn = nn.BatchNorm1d(input_dim + 1)
        
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

            combined_input = self.bn(combined_input)
            
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
    print("--- Test du DeepHedgingModel (Architecture Récurrente) ---")
    
    # 1. Paramètres de simulation
    batch_size = 64
    seq_len = 30     # 30 jours
    n_features = 3   # (Log-Price, Time-to-Maturity, Volatility)
    
    # 2. Création du modèle
    # Note : On déclare input_dim=3, mais le modèle sait qu'il doit attendre 4 entrées
    # (3 features marché + 1 position précédente) grâce à la modif dans __init__
    model = DeepHedgingModel(input_dim=n_features, hidden_dim=32, output_dim=1)
    
    # 3. Création d'une entrée aléatoire (Dummy data)
    # Shape : [Batch, Sequence Length, Features]
    dummy_input = torch.randn(batch_size, seq_len, n_features)
    
    # 4. Passage dans le modèle
    try:
        output = model(dummy_input)
        
        print(f"Input shape  : {dummy_input.shape}") # Doit être [64, 30, 3]
        print(f"Output shape : {output.shape}")      # Doit être [64, 30, 1]
        
        if output.shape == (batch_size, seq_len, 1):
            print("✅ SUCCÈS : Le modèle fonctionne techniquement.")
            print("   -> La boucle temporelle a correctement intégré le 'Delta Précédent' à chaque étape.")
        else:
            print("❌ ÉCHEC : Les dimensions de sortie ne sont pas celles attendues.")
            
    except RuntimeError as e:
        print(f"❌ ERREUR CRITIQUE pendant le forward : {e}")
        print("   -> Vérifie la ligne 'torch.cat' ou la taille de 'self.lstm_cell'.")