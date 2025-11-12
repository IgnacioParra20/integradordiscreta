from typing import List
from .activations import sigmoid, d_sigmoid_from_a
from .losses import bce

class MLP221:
    """
    MLP 2-2-1 didáctico (listas + for):
      - Sigmoide en oculta y salida
      - BCE
      - Pesos iniciales fijos (reproducible)
    """
    def __init__(self):
        # Entrada -> Oculta
        self.W1 = [[ 4.0,  4.0],   # h1
                   [-4.0, -4.0]]   # h2
        self.b1 = [-2.0, 6.0]

        # Oculta -> Salida
        self.W2 = [[6.0, 6.0]]
        self.b2 = [-9.0]

        # caches
        self.x  = [0.0, 0.0]
        self.z1 = [0.0, 0.0]
        self.a1 = [0.0, 0.0]
        self.z2 = [0.0]
        self.yhat = 0.0

        # gradientes
        self.dW1 = [[0.0, 0.0], [0.0, 0.0]]
        self.db1 = [0.0, 0.0]
        self.dW2 = [[0.0, 0.0]]
        self.db2 = [0.0]

    # ---------- Forward ----------
    def forward(self, x: List[float]) -> float:
        self.x = x[:]
        for o in range(2):
            z = self.b1[o]
            for i in range(2):
                z += self.W1[o][i] * x[i]
            self.z1[o] = z
            self.a1[o] = sigmoid(z)
        z = self.b2[0]
        for i in range(2):
            z += self.W2[0][i] * self.a1[i]
        self.z2[0] = z
        self.yhat = sigmoid(z)
        return self.yhat

    # ---------- Backward ----------
    def backward(self, y: float):
        # BCE + sigmoide: dL/dz2 = (yhat - y)
        delta2 = self.yhat - y

        for i in range(2):
            self.dW2[0][i] = delta2 * self.a1[i]
        self.db2[0] = delta2

        delta1 = [0.0, 0.0]
        for i in range(2):
            da = self.W2[0][i] * delta2
            delta1[i] = da * d_sigmoid_from_a(self.a1[i])

        for o in range(2):
            self.db1[o] = delta1[o]
            for i in range(2):
                self.dW1[o][i] = delta1[o] * self.x[i]

    # ---------- Update ----------
    def step(self, lr: float):
        for o in range(2):
            self.b1[o] -= lr * self.db1[o]
            for i in range(2):
                self.W1[o][i] -= lr * self.dW1[o][i]
        self.b2[0] -= lr * self.db2[0]
        for i in range(2):
            self.W2[0][i] -= lr * self.dW2[0][i]

    def predict(self, x: List[float]) -> float:
        return self.forward(x)
