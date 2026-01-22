import numpy as np
from scipy.stats import norm

class BlackScholesOracle:

    
    @staticmethod
    def get_price(S, K, T, r, sigma, option_type='call'):

        T = np.maximum(T, 1e-8)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type.lower() == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("option_type doit être 'call' ou 'put'")
            
        return price

    @staticmethod
    def get_delta(S, K, T, r, sigma, option_type='call'):

        T = np.maximum(T, 1e-8)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        if option_type.lower() == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

