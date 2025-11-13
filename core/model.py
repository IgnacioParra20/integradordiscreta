from typing import List
from .activations import sigmoid, d_sigmoid_from_a
from .losses import bce

class MLP221:
    """
    MLP didáctica con arquitectura 2-2-1 para el problema XOR.
    
    Arquitectura:
        - 2 neuronas de entrada (x1, x2)
        - 2 neuronas ocultas con activación sigmoide
        - 1 neurona de salida con activación sigmoide
        - Pérdida Binary Cross Entropy (BCE)
        
    La implementación usa listas y bucles de Python explícitos
    por claridad educativa en lugar de arreglos numpy.
    """
    def __init__(self):
        # Pesos y sesgos de la capa Entrada → Oculta
        # W1[i][j]: peso desde la entrada j a la neurona oculta i
        self.W1 = [[ 4.0,  4.0],   # pesos a h1
                   [-4.0, -4.0]]   # pesos a h2
        self.b1 = [-2.0, 6.0]      # sesgos para h1, h2

        # Pesos y sesgos de la capa Oculta → Salida
        # W2[0][i]: peso desde la neurona oculta i a la salida
        self.W2 = [[6.0, 6.0]]     # pesos desde h1, h2 a y
        self.b2 = [-9.0]           # sesgo para la salida y

        # Caché de la pasada hacia adelante (para visualización y backprop)
        self.x  = [0.0, 0.0]       # valores de entrada
        self.z1 = [0.0, 0.0]       # pre-activación de la capa oculta
        self.a1 = [0.0, 0.0]       # post-activación de la capa oculta
        self.z2 = [0.0]            # pre-activación de la salida
        self.yhat = 0.0            # salida predicha

        # Caché de gradientes
        self.dW1 = [[0.0, 0.0], [0.0, 0.0]]
        self.db1 = [0.0, 0.0]
        self.dW2 = [[0.0, 0.0]]
        self.db2 = [0.0]

    def forward(self, x: List[float]) -> float:
        """
        Propagación hacia adelante a través de la red.
        
        Args:
            x: Vector de entrada [x1, x2]
            
        Returns:
            Predicción de la red (yhat) después de la activación sigmoide
        """
        self.x = x[:]
        
        # Capa oculta: z1 = W1 @ x + b1, a1 = sigmoid(z1)
        for o in range(2):
            z = self.b1[o]
            for i in range(2):
                z += self.W1[o][i] * x[i]
            self.z1[o] = z
            self.a1[o] = sigmoid(z)
        
        # Capa de salida: z2 = W2 @ a1 + b2, yhat = sigmoid(z2)
        z = self.b2[0]
        for i in range(2):
            z += self.W2[0][i] * self.a1[i]
        self.z2[0] = z
        self.yhat = sigmoid(z)
        
        return self.yhat

    def backward(self, y: float):
        """
        Propagación hacia atrás para calcular gradientes.
        
        Para la pérdida BCE con salida sigmoide, el gradiente se simplifica a:
        dL/dz2 = yhat - y
        
        Args:
            y: Etiqueta verdadera (0 o 1)
        """
        # Gradiente de la capa de salida (BCE + derivada de sigmoide)
        delta2 = self.yhat - y

        # Gradientes para pesos y sesgo de la capa de salida
        for i in range(2):
            self.dW2[0][i] = delta2 * self.a1[i]
        self.db2[0] = delta2

        # Retropropagación a la capa oculta
        delta1 = [0.0, 0.0]
        for i in range(2):
            # Gradiente proveniente de la capa de salida
            da = self.W2[0][i] * delta2
            # Aplicar derivada de sigmoide
            delta1[i] = da * d_sigmoid_from_a(self.a1[i])

        # Gradientes para pesos y sesgos de la capa oculta
        for o in range(2):
            self.db1[o] = delta1[o]
            for i in range(2):
                self.dW1[o][i] = delta1[o] * self.x[i]

    def step(self, lr: float):
        """
        Actualiza pesos y sesgos usando los gradientes calculados.
        
        Args:
            lr: Tasa de aprendizaje para descenso de gradiente
        """
        # Actualizar parámetros de la capa oculta
        for o in range(2):
            self.b1[o] -= lr * self.db1[o]
            for i in range(2):
                self.W1[o][i] -= lr * self.dW1[o][i]
        
        # Actualizar parámetros de la capa de salida
        self.b2[0] -= lr * self.db2[0]
        for i in range(2):
            self.W2[0][i] -= lr * self.dW2[0][i]

    def predict(self, x: List[float]) -> float:
        """
        Realiza una predicción para la entrada x.
        
        Argumentos:
            x: Vector de entrada [x1, x2]
            
        Devuelve:
            Predicción de la red (igual que forward)
        """
        return self.forward(x)
