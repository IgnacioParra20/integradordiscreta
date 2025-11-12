from typing import List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core.model import MLP221
from data.xor import DATA

def export_loss_plot(losses, path="loss.png"):
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Época")
    plt.ylabel("Pérdida media (BCE)")
    plt.title("Curva de pérdida — XOR (MLP 2–2–1)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def export_pred_table(net: MLP221, path_md="predicciones.md"):
    with open(path_md, "w", encoding="utf-8") as f:
        f.write("# Predicciones XOR (final)\n\n")
        f.write("| x1 | x2 | y | ŷ |\n|---:|---:|---:|---:|\n")
        for x, y in DATA:
            yhat = net.predict(x)
            f.write(f"| {int(x[0])} | {int(x[1])} | {int(y)} | {yhat:.4f} |\n")
