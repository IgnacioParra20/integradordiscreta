import math

def sigmoid(z: float) -> float:
    """
    Sigmoid activation function with numerical stability.
    
    Uses different formulations depending on sign of z to avoid overflow:
    - For z >= 0: sigmoid(z) = 1 / (1 + exp(-z))
    - For z < 0:  sigmoid(z) = exp(z) / (1 + exp(z))
    
    Args:
        z: Pre-activation value
        
    Returns:
        Activation value in range (0, 1)
    """
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def d_sigmoid_from_a(a: float) -> float:
    """
    Derivative of sigmoid function given its output value.
    
    For sigmoid function σ(z), the derivative is:
    dσ/dz = σ(z) * (1 - σ(z)) = a * (1 - a)
    
    This formulation is more efficient since we typically already
    have the activation value a from the forward pass.
    
    Args:
        a: Sigmoid activation value (output of sigmoid)
        
    Returns:
        Derivative value
    """
    return a * (1.0 - a)
