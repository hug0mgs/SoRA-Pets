import math
import random

class PLDScheduler:
    """
    Implementa Progressive Layer Dropping (PLD) para Redes Neurais Transformers.
    Determina quais camadas devem ser desativadas ou excluídas durantes as épocas,
    incrementando a probabilidade de drop durante o progresso do treino e favoritando
    o dropping das camadas mais profundas.
    """
    def __init__(self, total_epochs, pld_limit, theta_bar=0.5):
        """
        PLD_Limit armazena até qual camada o PLD vai poder atuar.
        """
        self.total_epochs = total_epochs
        self.pld_limit = pld_limit
        self.theta_bar = theta_bar
        # Gamma controla a taxa de decaimento; 100/T é uma heurística que tirei do paper
        self.gamma = 100.0 / total_epochs if total_epochs > 0 else 0

    def get_drop_prob(self, current_epoch, layer_idx):
        """Calcula a probabilidade de dropping de uma camada específica"""
        # 1. Temporal Dimension: Overall survival rate drops exponentially
        theta_t = (1 - self.theta_bar) * math.exp(-self.gamma * current_epoch) + self.theta_bar
        
        # 2. Depth Dimension: Deeper layers (higher idx) have a higher chance to drop
        drop_prob = (layer_idx / self.pld_limit) * (1 - theta_t)
        return drop_prob

    def get_active_layers(self, current_epoch):
        """Retorna uma lista de índices de camadas (1-indexed) que continue ativa na época"""
        active_layers = []
        for i in range(1, self.pld_limit + 1):
            drop_prob = self.get_drop_prob(current_epoch, i)
            # If a random float is greater than the drop probability, the layer survives
            if random.random() > drop_prob:
                active_layers.append(i)
        return active_layers