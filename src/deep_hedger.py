import torch
import torch.nn as nn

class DeepHedgingModel(nn.Module):
    """
    Réseau de neurones récurrent pour le Deep Hedging.
    Architecture : LSTM + Couches Denses (MLP).
    Objectif : Prédire le Delta optimal à chaque pas de temps.
    """
    def __init__(self, input_dim=3, hidden_dim=32, num_layers=2, output_dim=1):
        super(DeepHedgingModel, self).__init__()
        
        # --- 1. Extraction de Features (Le "Cerveau" Récurrent) ---
        # Le LSTM prend la séquence temporelle et met à jour sa mémoire interne.
        # input_dim : Nombre de features (Prix, Temps restant, Volatilité...)
        # hidden_dim : Taille de la mémoire du réseau
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # --- 2. Prise de Décision (Le "Bras" qui agit) ---
        # Un petit réseau dense (MLP) qui prend la sortie du LSTM
        # et décide de l'action finale (Delta).
        self.decision_layer = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid() # Force la sortie entre 0 et 1 (car un Delta est souvent entre 0 et 100%)
        )

    def forward(self, x, prev_position=None):
        """
        Passe avant (Forward pass).
        
        Args:
            x (Tensor): État du marché [Batch, Sequence_Len, Features]
            prev_position (Tensor): (Optionnel) Position précédente pour gérer les coûts.
            
        Returns:
            deltas (Tensor): Séquence des positions de couverture [Batch, Sequence_Len, 1]
        """
        # 1. Passage dans le LSTM
        # out contient la "compréhension" du marché par le réseau à chaque pas de temps
        lstm_out, _ = self.lstm(x) 
        
        # 2. Passage dans le MLP décisionnel
        # On applique la décision à chaque étape temporelle
        deltas = self.decision_layer(lstm_out)
        
        return deltas

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