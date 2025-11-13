import math

def sigmoid(z: float) -> float:
    """
    Función de activación sigmoide con estabilidad numérica.
    
    Usa distintas formulaciones según el signo de z para evitar desbordamientos:
    - Para z >= 0: sigmoid(z) = 1 / (1 + exp(-z))
    - Para z < 0:  sigmoid(z) = exp(z) / (1 + exp(z))
    
    Parámetros:
        z: Valor pre-activación
        
    Devuelve:
        Valor de activación en el rango (0, 1)
    """
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def d_sigmoid_from_a(a: float) -> float:
    """
    Derivada de la función sigmoide dada su salida.

    Para la función sigmoide σ(z), la derivada es:
    dσ/dz = σ(z) * (1 - σ(z)) = a * (1 - a)

    Esta formulación es más eficiente ya que normalmente ya
    disponemos del valor de activación a durante la pasada hacia adelante.

    Parámetros:
        a: Valor de activación sigmoide (salida de la función sigmoide)

    Devuelve:
        Valor de la derivada
    """
    return a * (1.0 - a)
