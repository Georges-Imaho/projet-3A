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
    
    def simulate_heston(self, steps, n_paths=1, kappa=2.0, theta=0.2, xi=0.3, rho=-0.7):
        """
        Simule le modèle de Heston (Volatilité Stochastique).
        
        Args:
            kappa (float): Vitesse de retour à la moyenne de la volatilité.
            theta (float): Volatilité moyenne à long terme.
            xi (float): "Volatilité de la volatilité" (nervosité du marché).
            rho (float): Corrélation entre le prix et la volatilité (Souvent négative : quand le prix chute, la peur monte).
        
        Returns:
            paths (ndarray): Prix de l'actif [steps+1, n_paths]
            vol_paths (ndarray): Volatilité instantanée [steps+1, n_paths]
        """
        # 1. Génération des chocs aléatoires corrélés
        # Z1 pour le prix, Z2 pour la volatilité
        z1 = np.random.standard_normal((steps, n_paths))
        z2_uncorrelated = np.random.standard_normal((steps, n_paths))
        # Corrélation des browniens : W2 = rho*W1 + sqrt(1-rho^2)*Z2
        z2 = rho * z1 + np.sqrt(1 - rho**2) * z2_uncorrelated

        # 2. Initialisation des tableaux
        # On a besoin de stocker la volatilité à chaque instant
        price_paths = np.zeros((steps + 1, n_paths))
        vol_paths = np.zeros((steps + 1, n_paths))
        
        # Conditions initiales
        price_paths[0] = self.s0
        vol_paths[0] = theta # On commence à la moyenne long terme (ex: 20%)

        dt = self.dt
        sqrt_dt = np.sqrt(dt)

        # 3. Boucle temporelle (Euler-Maruyama)
        # On ne peut pas tout vectoriser d'un coup car t dépend de t-1
        for t in range(steps):
            S_t = price_paths[t]
            v_t = vol_paths[t]
            
            # S'assurer que la variance reste positive (Absorbtion ou Réflexion)
            v_t = np.maximum(v_t, 1e-5) # Sécurité numérique
            sqrt_vt = np.sqrt(v_t)

            # Mise à jour de la Volatilité (Processus CIR)
            # dv = kappa * (theta - v) * dt + xi * sqrt(v) * dW_vol
            d_vol = kappa * (theta - v_t) * dt + xi * sqrt_vt * z2[t] * sqrt_dt
            vol_paths[t+1] = v_t + d_vol
            
            # Mise à jour du Prix
            # dS = r * S * dt + sqrt(v) * S * dW_price
            d_price = self.r * S_t * dt + sqrt_vt * S_t * z1[t] * sqrt_dt
            price_paths[t+1] = S_t + d_price

        # On retourne les prix ET les volatilités (car le LSTM aura besoin de voir la vol)
        return price_paths, np.sqrt(np.maximum(vol_paths, 1e-5))

# Exemple d'utilisation rapide pour tester :
if __name__ == "__main__":
    sim = MarketSimulator(s0=100, r=0.05, sigma=0.2)
    # Simuler 10 trajectoires sur 252 jours (1 an)
    trajectoires = sim.simulate_gbm(steps=252, n_paths=10)
    print(f"Forme de la matrice de sortie : {trajectoires.shape}") # (253, 10)
    print(f"Prix finaux des 5 premières trajectoires : {trajectoires[-1, :5]}")