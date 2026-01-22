import numpy as np

class MarketSimulator:
    """
    Simulateur de prix d'actifs financiers.
    """
    def __init__(self, s0, r, sigma, dt=1/252):
        self.s0 = s0         
        self.r = r            
        self.sigma = sigma   
        self.dt = dt         
    def simulate_gbm(self, steps, n_paths=1):

        z = np.random.standard_normal((steps, n_paths))

        drift = (self.r - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * z

        log_returns = drift + diffusion
        cum_log_returns = np.cumsum(log_returns, axis=0)
        

        paths = self.s0 * np.exp(np.vstack([np.zeros(n_paths), cum_log_returns]))
        
        return paths
    
    def simulate_heston(self, steps, n_paths=1, kappa=2.0, theta=0.2, xi=0.3, rho=-0.7):

        z1 = np.random.standard_normal((steps, n_paths))
        z2_uncorrelated = np.random.standard_normal((steps, n_paths))

        z2 = rho * z1 + np.sqrt(1 - rho**2) * z2_uncorrelated

        price_paths = np.zeros((steps + 1, n_paths))
        vol_paths = np.zeros((steps + 1, n_paths))

        price_paths[0] = self.s0
        vol_paths[0] = theta
        dt = self.dt
        sqrt_dt = np.sqrt(dt)

        for t in range(steps):
            S_t = price_paths[t]
            v_t = vol_paths[t]

            v_t = np.maximum(v_t, 1e-5)
            sqrt_vt = np.sqrt(v_t)

            d_vol = kappa * (theta - v_t) * dt + xi * sqrt_vt * z2[t] * sqrt_dt
            vol_paths[t+1] = v_t + d_vol

            d_price = self.r * S_t * dt + sqrt_vt * S_t * z1[t] * sqrt_dt
            price_paths[t+1] = S_t + d_price

        return price_paths, np.sqrt(np.maximum(vol_paths, 1e-5))
