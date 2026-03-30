import numpy as np
from abc import ABC, abstractmethod
from typing import List

class Agent(ABC):
    """Interface para todos os agentes."""
    @abstractmethod
    def predict(self, state: np.ndarray) -> int:
        """Faz uma previsão de ação com base no estado atual."""
        pass

class HumanAgent(Agent):
    """Agente controlado por um humano (para modo manual)"""
    def predict(self, state: np.ndarray) -> int:
        # O estado é ignorado - entrada vem do teclado
        return 0  # Padrão: não fazer nada (será sobrescrito pela entrada do usuário no manual_play.py)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exps = np.exp(x - np.max(x))  #estabilidade numérica
    return exps / np.sum(exps)

class NeuralNetworkAgent(Agent):
    def __init__(self, weights_path: str = None):
        if weights_path is not None:
            self.weights = np.load(weights_path, allow_pickle=True).item()
        else:
            self.weights = {}
    
    def predict(self, state: np.ndarray) -> int:
        x = state

        z1 = np.dot(self.weights["W1"], x) + self.weights["b1"]
        a1 = tanh(z1)

        z2 = np.dot(self.weights["W2"], a1) + self.weights["b2"]
        a2 = tanh(z2)

        z3 = np.dot(self.weights["W3"], a2) + self.weights["b3"]
        output = softmax(z3)

        action = np.argmax(output)
        return action

