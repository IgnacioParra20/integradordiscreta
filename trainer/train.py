from typing import Optional
from core.model import MLP221
from core.losses import bce
from data.xor import DATA
from mlpio.tracer import MarkdownTracer

def train(net: MLP221, epochs: int = 3000, lr: float = 0.5, tracer: Optional[MarkdownTracer] = None):
    losses = []
    for ep in range(1, epochs + 1):
        if tracer: tracer.log_epoch_header(ep, lr)
        ep_loss = 0.0
        for x, y in DATA:
            yhat = net.forward(x)
            L = bce(yhat, y)
            ep_loss += L
            net.backward(y)
            if tracer: tracer.log_sample(x, y, net)
            net.step(lr)
            if tracer: tracer.log_update(net)
        losses.append(ep_loss / len(DATA))
    return losses
