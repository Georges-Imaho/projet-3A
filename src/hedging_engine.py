import torch
import torch.nn as nn
import numpy as np

class HedgingEngine:
    def __init__(self, model, optimizer, criterion, transaction_cost_pct=0.0, risk_aversion=1.0):
        self.model = model
        self.optimizer = optimizer
        # self.criterion n'est plus utilisé directement, on gère en interne
        self.cost_pct = transaction_cost_pct
        self.risk_aversion = risk_aversion

    # ... (Garde ta méthode _compute_pnl et entropic_loss EXACTEMENT comme avant) ...
    def _compute_pnl(self, spot_paths, strikes, deltas, T, initial_prices=None):
        # ... (Ton code précédent pour _compute_pnl) ...
        # (Copie-colle le bloc précédent ici, je l'abrége pour la clarté)
        price_changes = torch.diff(spot_paths, dim=1)
        active_deltas = deltas[:, :-1, 0]
        hedging_pnl = torch.sum(active_deltas * price_changes, dim=1)
        zeros = torch.zeros((deltas.shape[0], 1, 1), device=deltas.device)
        padded_deltas = torch.cat([zeros, deltas], dim=1)
        delta_changes = torch.abs(torch.diff(padded_deltas, dim=1))
        costs = torch.sum(delta_changes[:, :-1, 0] * spot_paths[:, :-1] * self.cost_pct, dim=1)
        final_prices = spot_paths[:, -1]
        option_payoff = torch.relu(final_prices - strikes)
        if initial_prices is None: initial_prices = torch.zeros_like(option_payoff)
        return initial_prices + hedging_pnl - costs - option_payoff

    def entropic_loss(self, pnl):
        x = -self.risk_aversion * pnl
        # Clamp pour éviter l'explosion (Sécurité ultime)
        x = torch.clamp(x, max=50.0) 
        log_sum_exp = torch.logsumexp(x, dim=0)
        n = torch.tensor(x.size(0), device=x.device, dtype=x.dtype)
        return (log_sum_exp - torch.log(n)) / self.risk_aversion

    def train_step(self, spot_paths, strikes, inputs, initial_prices, use_mse=False):
        self.model.train()
        self.optimizer.zero_grad()
        
        deltas = self.model(inputs)
        pnl = self._compute_pnl(spot_paths, strikes, deltas, T=1.0, initial_prices=initial_prices)
        
        if torch.isnan(pnl).any():
            pnl = torch.nan_to_num(pnl, nan=0.0)

        # --- LE CŒUR DE LA CORRECTION ---
        if use_mse:
            # Mode "Échauffement" : On veut juste que PnL soit proche de 0
            # C'est très stable numériquement.
            loss = torch.mean(pnl**2)
        else:
            # Mode "Performance" : Une fois stable, on optimise l'utilité
            loss = self.entropic_loss(pnl)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5) # Clip plus strict
        self.optimizer.step()
        
        return loss.item(), torch.mean(pnl).item()
    
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