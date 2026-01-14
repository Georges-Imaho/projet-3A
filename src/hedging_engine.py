import torch
import torch.nn as nn
import numpy as np

class HedgingEngine:
    """
    Le Gymnase : Gère la simulation, le calcul du PnL (Profits & Pertes) 
    et l'apprentissage du modèle avec prise en compte des frictions.
    """
    def __init__(self, model, optimizer, criterion, transaction_cost_pct=0.0):
        """
        Args:
            model: Le réseau de neurones (DeepHedger)
            optimizer: L'optimiseur (Adam)
            criterion: La fonction de perte de base (souvent MSE)
            transaction_cost_pct (float): Coûts de transaction (ex: 0.001 pour 0.1%)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.cost_pct = transaction_cost_pct

    def _compute_pnl(self, spot_paths, strikes, deltas, T):
        """
        Calcule le PnL (Profit and Loss) final de la stratégie de couverture.
        C'est le calcul financier pur.
        """
        # 1. Calcul des variations de prix (dS) : S(t+1) - S(t)
        # spot_paths shape: [Batch, Time]
        price_changes = torch.diff(spot_paths, dim=1) 
        
        # 2. Alignement des deltas
        # On utilise le delta décidé en t pour profiter du mouvement entre t et t+1
        # On coupe le dernier delta car il ne sert à rien à la maturité
        active_deltas = deltas[:, :-1, 0] 
        
        # 3. Profit généré par la stratégie de hedging (Gain sur actions)
        # Somme (Quantité détenue * Variation de prix)
        hedging_pnl = torch.sum(active_deltas * price_changes, dim=1)
        
        # 4. Calcul des coûts de transaction
        # Coût = |Delta_t - Delta_{t-1}| * Prix_t * Taux_Frais
        # On ajoute une colonne de zéros au début pour le premier achat
        zeros = torch.zeros((deltas.shape[0], 1, 1), device=deltas.device)
        padded_deltas = torch.cat([zeros, deltas], dim=1)
        
        # Changement de position
        delta_changes = torch.abs(torch.diff(padded_deltas, dim=1))
        
        # On simplifie en appliquant les frais sur le prix moyen ou spot instantané
        # Ici on applique sur le spot path aligné
        costs = torch.sum(delta_changes[:, :-1, 0] * spot_paths[:, :-1] * self.cost_pct, dim=1)
        
        # 5. Payoff de l'option (Ce qu'on doit payer au client à la fin)
        # Payoff Call = Max(S_T - K, 0)
        final_prices = spot_paths[:, -1]
        option_payoff = torch.relu(final_prices - strikes)
        
        # 6. PnL Total = Premium (vendu au début) + Hedging PnL - Costs - Payoff (payé à la fin)
        # Note : Pour l'entraînement, on cherche juste à minimiser la variance entre 
        # (Hedging - Costs) et (Payoff). On ignore souvent le Premium car c'est une constante.
        
        return hedging_pnl - costs - option_payoff

    def train_step(self, spot_paths, strikes, inputs):
        """
        Une étape d'entraînement avec la nouvelle Loss Function.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 1. L'IA décide
        deltas = self.model(inputs)
        
        # 2. On calcule le résultat financier
        pnl = self._compute_pnl(spot_paths, strikes, deltas, T=1.0)
        
        # 3. NOUVELLE LOSS FUNCTION (Mean-Variance Optimization)
        # On veut :
        #   - Minimiser le Risque (Variance)
        #   - Maximiser le Gain (Moyenne)
        # Donc on minimise : Variance - lambda * Moyenne
        
        risk = torch.var(pnl)      # Risque (Stabilité)
        reward = torch.mean(pnl)   # Profit (Économies de frais)
        
        # La formule magique :
        loss = risk - (self.risk_aversion * reward)
        
        # 4. Optimisation
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), reward.item()

# --- Test rapide de la logique de calcul ---
if __name__ == "__main__":
    # Simulation de données factices
    batch_sz = 5
    seq_len = 10
    
    # Prix qui montent de 100 à 110
    spots = torch.linspace(100, 110, seq_len).repeat(batch_sz, 1)
    strikes = torch.full((batch_sz,), 100.0)
    
    # Deltas factices (Le modèle achète 0.5 action et garde)
    deltas = torch.full((batch_sz, seq_len, 1), 0.5)
    
    # Instance Engine (sans modèle pour tester juste la méthode statique PnL)
    engine = HedgingEngine(None, None, None, transaction_cost_pct=0.01)
    
    # Calcul PnL
    pnl = engine._compute_pnl(spots, strikes, deltas, T=1.0)
    
    print(f"PnL Batch shape : {pnl.shape}")
    print(f"PnL Moyen : {pnl.mean().item():.2f}")
    print("✅ Le moteur de calcul financier fonctionne.")