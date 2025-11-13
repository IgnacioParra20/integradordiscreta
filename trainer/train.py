from typing import Optional, Callable
from core.model import MLP221
from core.losses import bce
from data.xor import DATA
from mlpio.tracer import MarkdownTracer

def train(net: MLP221, epochs: int = 3000, lr: float = 0.5, tracer: Optional[MarkdownTracer] = None):
    """
    Standard training loop without callbacks.
    
    Args:
        net: The MLP221 model to train
        epochs: Number of training epochs
        lr: Learning rate
        tracer: Optional tracer for logging training details
        
    Returns:
        List of average losses per epoch
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
    Training loop with callback support for real-time UI updates.
    
    Args:
        net: The MLP221 model to train
        epochs: Number of training epochs
        lr: Learning rate
        tracer: Optional tracer for logging training details
        callback: Optional callback function(epoch, total_epochs, avg_loss)
        
    Returns:
        List of average losses per epoch
    """
    losses = []
    for ep in range(1, epochs + 1):
        if tracer: 
            tracer.log_epoch_header(ep, lr)
        ep_loss = 0.0
        
        # Train on all samples
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
        
        # Calculate average loss for this epoch
        avg_loss = ep_loss / len(DATA)
        losses.append(avg_loss)
        
        # Invoke callback for UI updates
        if callback:
            callback(ep, epochs, avg_loss)
    
    return losses
