import math

def bce(yhat: float, y: float, eps: float = 1e-12) -> float:
    """
    Binary Cross Entropy loss function.
    
    BCE(ŷ, y) = -[y * log(ŷ) + (1-y) * log(1-ŷ)]
    
    Includes epsilon clipping to prevent log(0) errors.
    
    Args:
        yhat: Predicted probability in range [0, 1]
        y: True label (0 or 1)
        eps: Small constant to prevent numerical instability
        
    Returns:
        BCE loss value
    """
    # Clip predictions to avoid log(0)
    yhat = min(max(yhat, eps), 1.0 - eps)
    return -(y * math.log(yhat) + (1.0 - y) * math.log(1.0 - yhat))
