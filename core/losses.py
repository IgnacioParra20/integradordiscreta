import math

def bce(yhat: float, y: float, eps: float = 1e-12) -> float:
    """
    Función de pérdida de entropía cruzada binaria.
    
    BCE(ŷ, y) = -[y * log(ŷ) + (1-y) * log(1-ŷ)]
    
    Incluye recorte con epsilon para evitar errores por log(0).
    
    Parámetros:
        yhat: Probabilidad predicha en el rango [0, 1]
        y: Etiqueta verdadera (0 o 1)
        eps: Pequeña constante para prevenir inestabilidad numérica
        
    Devuelve:
        Valor de la pérdida BCE
    """
    # Recortar predicciones para evitar log(0)
    yhat = min(max(yhat, eps), 1.0 - eps)
    return -(y * math.log(yhat) + (1.0 - y) * math.log(1.0 - yhat))
