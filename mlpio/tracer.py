from typing import List, Tuple
from core.model import MLP221
from core.losses import bce

class MarkdownTracer:
    def __init__(self, path_md: str = "trazas.md"):
        self.path = path_md
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("# Trazas MLP 2–2–1 (XOR) — BCE + Sigmoide\n\n")

    def log_epoch_header(self, epoch: int, lr: float):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"\n---\n\n## Época {epoch} (lr={lr})\n\n")

    def log_sample(self, x, y, net: MLP221):
        L = bce(net.yhat, y)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"**Entrada** `x={x}`, **y**=`{y}`\n\n")
            f.write("**Pesos antes**\n\n")
            f.write(f"- W1={net.W1}  \n- b1={net.b1}  \n- W2={net.W2}  \n- b2={net.b2}\n\n")
            f.write("**Forward**\n\n")
            f.write(f"- z1={net.z1}  \n- a1={net.a1}  \n- z2={net.z2}  \n- yhat={net.yhat:.6f}\n\n")
            f.write(f"**Pérdida BCE**: `{L:.6f}`\n\n")
            f.write("**Gradientes**\n\n")
            f.write(f"- dW2={net.dW2}  \n- db2={net.db2}  \n- dW1={net.dW1}  \n- db1={net.db1}\n\n")

    def log_update(self, net: MLP221):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("**Pesos después del update**\n\n")
            f.write(f"- W1={net.W1}  \n- b1={net.b1}  \n- W2={net.W2}  \n- b2={net.b2}\n\n")

    def log_final_predictions(self, preds):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("\n---\n\n## Predicciones finales\n\n")
            f.write("| x1 | x2 | y | ŷ |\n|---:|---:|---:|---:|\n")
            for (x,y), yhat in preds:
                f.write(f"| {int(x[0])} | {int(x[1])} | {int(y)} | {yhat:.4f} |\n")
            f.write("\n")
