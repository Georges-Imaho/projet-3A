import numpy as np

class MarketSimulator:
    """
    Simulateur de prix d'actifs financiers.
    """
    def __init__(self, s0, r, sigma, dt=1/252):
        self.s0 = s0          # Prix initial de l'action
        self.r = r            # Taux sans risque (ex: 0.03 pour 3%)
        self.sigma = sigma    # Volatilité (ex: 0.2 pour 20%)
        self.dt = dt          # Pas de temps (1/252 = 1 jour de bourse)

    def simulate_gbm(self, steps, n_paths=1):
        """
        Génère des trajectoires via le Mouvement Brownien Géométrique (GBM).
        Formule : dS_t = r*S_t*dt + sigma*S_t*dW_t
        """
        # Génération des chocs aléatoires normaux (dW_t)
        # Taille : (Nombre de pas, Nombre de trajectoires)
        z = np.random.standard_normal((steps, n_paths))
        
        # Calcul de la dérive et de la diffusion
        # On utilise la forme exponentielle : S_t = S_0 * exp((r - 0.5*sigma^2)*t + sigma*W_t)
        drift = (self.r - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * z
        
        # Calcul des rendements logarithmiques cumulés
        log_returns = drift + diffusion
        cum_log_returns = np.cumsum(log_returns, axis=0)
        
        # Ajout du prix initial (log(S0)) et passage à l'exponentielle
        paths = self.s0 * np.exp(np.vstack([np.zeros(n_paths), cum_log_returns]))
        
        return paths

# Exemple d'utilisation rapide pour tester :
if __name__ == "__main__":
    sim = MarketSimulator(s0=100, r=0.05, sigma=0.2)
    # Simuler 10 trajectoires sur 252 jours (1 an)
    trajectoires = sim.simulate_gbm(steps=252, n_paths=10)
    print(f"Forme de la matrice de sortie : {trajectoires.shape}") # (253, 10)
    print(f"Prix finaux des 5 premières trajectoires : {trajectoires[-1, :5]}")