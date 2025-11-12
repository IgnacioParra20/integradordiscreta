import math

def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

def d_sigmoid_from_a(a: float) -> float:
    return a * (1.0 - a)
