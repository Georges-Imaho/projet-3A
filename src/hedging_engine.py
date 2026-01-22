import torch
import torch.nn as nn

class HedgingEngine:
    def __init__(self, model, optimizer, criterion, transaction_cost_pct=0.0, risk_aversion=1.0):
        self.model = model
        self.optimizer = optimizer
        self.cost_pct = transaction_cost_pct
        self.risk_aversion = risk_aversion
        self.warmup_criterion = nn.MSELoss() 
    def _compute_pnl(self, spot_paths, strikes, deltas, T, initial_prices=None):

        price_changes = torch.diff(spot_paths, dim=1)
        active_deltas = deltas[:, :-1, 0]
        hedging_pnl = torch.sum(active_deltas * price_changes, dim=1)
        
        zeros = torch.zeros((deltas.shape[0], 1, 1), device=deltas.device)
        padded_deltas = torch.cat([zeros, deltas], dim=1)
        delta_changes = torch.abs(torch.diff(padded_deltas, dim=1))

        costs = torch.sum(delta_changes[:, :-1, 0] * spot_paths[:, :-1] * self.cost_pct, dim=1)
        
        final_prices = spot_paths[:, -1]
        option_payoff = torch.relu(final_prices - strikes)
        
        if initial_prices is None:
            initial_prices = torch.zeros_like(option_payoff)

        pnl = initial_prices + hedging_pnl - costs - option_payoff
        return pnl

    def entropic_loss(self, pnl):

        pnl_normalized = pnl / 100.0 
        
        x = -self.risk_aversion * pnl_normalized

        log_sum_exp = torch.logsumexp(x, dim=0)
        n = torch.tensor(x.size(0), device=x.device, dtype=x.dtype)
        
        return (log_sum_exp - torch.log(n))

    def train_step(self, spot_paths, strikes, inputs, initial_prices, use_mse=False):
        self.model.train()
        self.optimizer.zero_grad()
        
        deltas = self.model(inputs)
        pnl = self._compute_pnl(spot_paths, strikes, deltas, T=1.0, initial_prices=initial_prices)

        if torch.isnan(pnl).any(): pnl = torch.nan_to_num(pnl, nan=0.0)

        if use_mse:

            loss = torch.mean(pnl**2)
        else:
            loss = self.entropic_loss(pnl)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item(), torch.mean(pnl).item()
