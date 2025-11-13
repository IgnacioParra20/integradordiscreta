from typing import List
from .activations import sigmoid, d_sigmoid_from_a
from .losses import bce

class MLP221:
    """
    Didactic MLP with 2-2-1 architecture for XOR problem.
    
    Architecture:
        - 2 input neurons (x1, x2)
        - 2 hidden neurons with sigmoid activation
        - 1 output neuron with sigmoid activation
        - Binary Cross Entropy (BCE) loss
        
    The implementation uses explicit Python lists and loops
    for educational clarity rather than numpy arrays.
    """
    def __init__(self):
        # Input → Hidden layer weights and biases
        # W1[i][j]: weight from input j to hidden neuron i
        self.W1 = [[ 4.0,  4.0],   # weights to h1
                   [-4.0, -4.0]]   # weights to h2
        self.b1 = [-2.0, 6.0]      # biases for h1, h2

        # Hidden → Output layer weights and biases
        # W2[0][i]: weight from hidden neuron i to output
        self.W2 = [[6.0, 6.0]]     # weights from h1, h2 to y
        self.b2 = [-9.0]           # bias for output y

        # Forward pass cache (for visualization and backprop)
        self.x  = [0.0, 0.0]       # input values
        self.z1 = [0.0, 0.0]       # pre-activation hidden layer
        self.a1 = [0.0, 0.0]       # post-activation hidden layer
        self.z2 = [0.0]            # pre-activation output
        self.yhat = 0.0            # predicted output

        # Gradient cache
        self.dW1 = [[0.0, 0.0], [0.0, 0.0]]
        self.db1 = [0.0, 0.0]
        self.dW2 = [[0.0, 0.0]]
        self.db2 = [0.0]

    def forward(self, x: List[float]) -> float:
        """
        Forward propagation through the network.
        
        Args:
            x: Input vector [x1, x2]
            
        Returns:
            Network prediction (yhat) after sigmoid activation
        """
        self.x = x[:]
        
        # Hidden layer: z1 = W1 @ x + b1, a1 = sigmoid(z1)
        for o in range(2):
            z = self.b1[o]
            for i in range(2):
                z += self.W1[o][i] * x[i]
            self.z1[o] = z
            self.a1[o] = sigmoid(z)
        
        # Output layer: z2 = W2 @ a1 + b2, yhat = sigmoid(z2)
        z = self.b2[0]
        for i in range(2):
            z += self.W2[0][i] * self.a1[i]
        self.z2[0] = z
        self.yhat = sigmoid(z)
        
        return self.yhat

    def backward(self, y: float):
        """
        Backward propagation to compute gradients.
        
        For BCE loss with sigmoid output, the gradient simplifies to:
        dL/dz2 = yhat - y
        
        Args:
            y: True label (0 or 1)
        """
        # Output layer gradient (BCE + sigmoid derivative)
        delta2 = self.yhat - y

        # Gradients for output layer weights and bias
        for i in range(2):
            self.dW2[0][i] = delta2 * self.a1[i]
        self.db2[0] = delta2

        # Backpropagate to hidden layer
        delta1 = [0.0, 0.0]
        for i in range(2):
            # Gradient from output layer
            da = self.W2[0][i] * delta2
            # Apply sigmoid derivative
            delta1[i] = da * d_sigmoid_from_a(self.a1[i])

        # Gradients for hidden layer weights and biases
        for o in range(2):
            self.db1[o] = delta1[o]
            for i in range(2):
                self.dW1[o][i] = delta1[o] * self.x[i]

    def step(self, lr: float):
        """
        Update weights and biases using computed gradients.
        
        Args:
            lr: Learning rate for gradient descent
        """
        # Update hidden layer parameters
        for o in range(2):
            self.b1[o] -= lr * self.db1[o]
            for i in range(2):
                self.W1[o][i] -= lr * self.dW1[o][i]
        
        # Update output layer parameters
        self.b2[0] -= lr * self.db2[0]
        for i in range(2):
            self.W2[0][i] -= lr * self.dW2[0][i]

    def predict(self, x: List[float]) -> float:
        """
        Make a prediction for input x.
        
        Args:
            x: Input vector [x1, x2]
            
        Returns:
            Network prediction (same as forward)
        """
        return self.forward(x)
