import math

def bce(yhat: float, y: float, eps: float = 1e-12) -> float:
    yhat = min(max(yhat, eps), 1.0 - eps)
    return -(y * math.log(yhat) + (1.0 - y) * math.log(1.0 - yhat))
