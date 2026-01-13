import numpy as np
from scipy.stats import norm

class BlackScholesOracle:
    """
    L'Oracle : Implémente la formule fermée de Black-Scholes pour le pricing.
    Supporte la vectorisation (entrées sous forme de tableaux NumPy).
    """
    
    @staticmethod
    def get_price(S, K, T, r, sigma, option_type='call'):
        """
        Calcule le prix d'une option européenne.
        
        Args:
            S (float or array): Prix actuel du sous-jacent (Spot)
            K (float or array): Prix d'exercice (Strike)
            T (float or array): Temps restant jusqu'à maturité (en années)
            r (float or array): Taux d'intérêt sans risque
            sigma (float or array): Volatilité du sous-jacent
            option_type (str): 'call' ou 'put'
            
        Returns:
            float or array: Prix de l'option
        """
        # Pour éviter la division par zéro si T=0
        T = np.maximum(T, 1e-8)
        
        # Calcul de d1 et d2
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
        """
        Calcule le Delta de l'option (utile pour la comparaison en Semaine 3).
        Delta = dV/dS (sensibilité du prix de l'option au prix du sous-jacent).
        """
        T = np.maximum(T, 1e-8)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        if option_type.lower() == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

# --- Petit script de test ---
if __name__ == "__main__":
    # Test unitaire
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    
    call_price = BlackScholesOracle.get_price(S, K, T, r, sigma, 'call')
    put_price = BlackScholesOracle.get_price(S, K, T, r, sigma, 'put')
    delta_call = BlackScholesOracle.get_delta(S, K, T, r, sigma, 'call')
    
    print(f"Test pour S=100, K=100, T=1an, r=5%, sigma=20%:")
    print(f"  - Prix du Call : {call_price:.4f}")
    print(f"  - Prix du Put  : {put_price:.4f}")
    print(f"  - Delta du Call: {delta_call:.4f}")

    # Test de vectorisation (calculer 3 prix d'un coup)
    S_vec = np.array([90, 100, 110])
    prices = BlackScholesOracle.get_price(S_vec, K, T, r, sigma)
    print(f"\nTest vectorisation (3 prix) : {prices}")