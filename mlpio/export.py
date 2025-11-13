"""Utilidades para exportar resultados del entrenamiento a archivos."""

from typing import Iterable

import matplotlib

# Seleccionar backend sin GUI para poder guardar gráficos en cualquier entorno.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from core.model import MLP221
from data.xor import DATA


def export_loss_plot(losses: Iterable[float], path: str = "loss.png") -> None:
    """Genera y guarda la curva de pérdida promedio por época."""

    loss_values = list(losses)
    plt.figure()
    plt.plot(range(1, len(loss_values) + 1), loss_values)
    plt.xlabel("Época")
    plt.ylabel("Pérdida media (BCE)")
    plt.title("Curva de pérdida — XOR (MLP 2–2–1)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def export_pred_table(net: MLP221, path_md: str = "predicciones.md") -> None:
    """Escribe una tabla Markdown con las predicciones actuales de la red."""

    with open(path_md, "w", encoding="utf-8") as file:
        file.write("# Predicciones XOR (final)\n\n")
        file.write("| x1 | x2 | y | ŷ |\n|---:|---:|---:|---:|\n")
        for x, y in DATA:
            yhat = net.predict(x)
            file.write(f"| {int(x[0])} | {int(x[1])} | {int(y)} | {yhat:.4f} |\n")
