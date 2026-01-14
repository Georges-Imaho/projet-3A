#### REMPLACE TOUT LE CONTENU DE : src/hedging_engine.py ####
import torch
import torch.nn as nn
import numpy as np

class HedgingEngine:
    def __init__(self, model, optimizer, criterion, transaction_cost_pct=0.0, risk_aversion=1.0):
        self.model = model
        self.optimizer = optimizer
        self.cost_pct = transaction_cost_pct
        self.risk_aversion = risk_aversion
        # HuberLoss est beaucoup plus stable que MSE pour le début
        self.warmup_criterion = nn.SmoothL1Loss() 

    def _compute_pnl(self, spot_paths, strikes, deltas, T, initial_prices=None):
        price_changes = torch.diff(spot_paths, dim=1)
        active_deltas = deltas[:, :-1, 0]
        
        hedging_pnl = torch.sum(active_deltas * price_changes, dim=1)
        
        # Calcul des frais de transaction
        zeros = torch.zeros((deltas.shape[0], 1, 1), device=deltas.device)
        padded_deltas = torch.cat([zeros, deltas], dim=1)
        delta_changes = torch.abs(torch.diff(padded_deltas, dim=1))
        costs = torch.sum(delta_changes[:, :-1, 0] * spot_paths[:, :-1] * self.cost_pct, dim=1)
        
        final_prices = spot_paths[:, -1]
        option_payoff = torch.relu(final_prices - strikes)
        
        if initial_prices is None:
            initial_prices = torch.zeros_like(option_payoff)
            
        return initial_prices + hedging_pnl - costs - option_payoff

    def entropic_loss(self, pnl):
        # 1. Scaling pour éviter l'overflow
        # Si le PnL est trop grand, l'exponentielle explose. On clamp.
        x = -self.risk_aversion * pnl
        x = torch.clamp(x, max=80.0) # exp(80) est la limite float32 safe
        
        # 2. LogSumExp Trick
        log_sum_exp = torch.logsumexp(x, dim=0)
        n = torch.tensor(x.size(0), device=x.device, dtype=x.dtype)
        
        return (log_sum_exp - torch.log(n)) / self.risk_aversion

    def train_step(self, spot_paths, strikes, inputs, initial_prices, use_mse=False):
        self.model.train()
        self.optimizer.zero_grad()
        
        deltas = self.model(inputs)
        pnl = self._compute_pnl(spot_paths, strikes, deltas, T=1.0, initial_prices=initial_prices)
        
        # Sécurité Anti-NaN
        if torch.isnan(pnl).any():
            pnl = torch.nan_to_num(pnl, nan=0.0)

        if use_mse:
            # On utilise SmoothL1 au lieu de MSE pure pour éviter l'explosion
            # On vise PnL = 0 (Perfect Hedge)
            target = torch.zeros_like(pnl)
            loss = self.warmup_criterion(pnl, target)
        else:
            loss = self.entropic_loss(pnl)
        
        if torch.isnan(loss):
            return 0.0, 0.0 # On skip le step si NaN
            
        loss.backward()
        
        # Clipping très agressif pour forcer la stabilité
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
        
        self.optimizer.step()
        
        return loss.item(), torch.mean(pnl).item()