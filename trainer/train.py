from typing import Optional, Callable
from core.model import MLP221
from core.losses import bce
from data.xor import DATA
from mlpio.tracer import MarkdownTracer

def train(net: MLP221, epochs: int = 3000, lr: float = 0.5, tracer: Optional[MarkdownTracer] = None):
    """
    Bucle de entrenamiento estándar sin callbacks.
    
    Argumentos:
        net: El modelo MLP221 a entrenar
        epochs: Número de épocas de entrenamiento
        lr: Tasa de aprendizaje
        tracer: Trazador opcional para registrar detalles del entrenamiento
        
    Devuelve:
        Lista de pérdidas promedio por época
    """
    losses = []
    for ep in range(1, epochs + 1):
        if tracer: 
            tracer.log_epoch_header(ep, lr)
        ep_loss = 0.0
        for x, y in DATA:
            yhat = net.forward(x)
            L = bce(yhat, y)
            ep_loss += L
            net.backward(y)
            if tracer: 
                tracer.log_sample(x, y, net)
            net.step(lr)
            if tracer: 
                tracer.log_update(net)
        losses.append(ep_loss / len(DATA))
    return losses


def train_with_callback(
    net: MLP221, 
    epochs: int = 3000, 
    lr: float = 0.5, 
    tracer: Optional[MarkdownTracer] = None,
    callback: Optional[Callable[[int, int, float], None]] = None
):
    """
    Bucle de entrenamiento con soporte de callback para actualizaciones en tiempo real de la interfaz.
    
    Argumentos:
        net: El modelo MLP221 a entrenar
        epochs: Número de épocas de entrenamiento
        lr: Tasa de aprendizaje
        tracer: Trazador opcional para registrar detalles del entrenamiento
        callback: Función callback opcional(epoch, total_epochs, avg_loss)
        
    Devuelve:
        Lista de pérdidas promedio por época
    """
    losses = []
    for ep in range(1, epochs + 1):
        if tracer: 
            tracer.log_epoch_header(ep, lr)
        ep_loss = 0.0
        
        # Entrenar con todas las muestras
        for x, y in DATA:
            yhat = net.forward(x)
            L = bce(yhat, y)
            ep_loss += L
            net.backward(y)
            if tracer: 
                tracer.log_sample(x, y, net)
            net.step(lr)
            if tracer: 
                tracer.log_update(net)
        
        # Calcular la pérdida promedio para esta época
        avg_loss = ep_loss / len(DATA)
        losses.append(avg_loss)
        
        # Invocar callback para actualizaciones de la UI
        if callback:
            callback(ep, epochs, avg_loss)
    
    return losses
