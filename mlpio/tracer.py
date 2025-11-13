"""Herramientas para registrar el entrenamiento en archivos Markdown."""

from typing import Iterable, Sequence, Tuple

from core.losses import bce
from core.model import MLP221


class MarkdownTracer:
    """Genera bitácoras con todo el detalle numérico del entrenamiento.

    Cada llamada abre el archivo de salida en modo append para que la traza se
    mantenga en orden cronológico y sea fácil de leer durante clases o talleres.
    """

    def __init__(self, path_md: str = "trazas.md"):
        """Crea el archivo Markdown con un encabezado introductorio."""

        self.path = path_md
        with open(self.path, "w", encoding="utf-8") as file:
            file.write("# Trazas MLP 2–2–1 (XOR) — BCE + Sigmoide\n\n")

    def log_epoch_header(self, epoch: int, lr: float) -> None:
        """Inserta un separador y encabezado para cada época registrada."""

        with open(self.path, "a", encoding="utf-8") as file:
            file.write(f"\n---\n\n## Época {epoch} (lr={lr})\n\n")

    def log_sample(self, x, y, net: MLP221) -> None:
        """Captura los valores antes/después de la pasada forward/backward."""

        loss = bce(net.yhat, y)
        with open(self.path, "a", encoding="utf-8") as file:
            file.write(f"**Entrada** `x={x}`, **y**=`{y}`\n\n")
            file.write("**Pesos antes**\n\n")
            file.write(
                f"- W1={net.W1}  \n- b1={net.b1}  \n- W2={net.W2}  \n- b2={net.b2}\n\n"
            )
            file.write("**Forward**\n\n")
            file.write(
                f"- z1={net.z1}  \n- a1={net.a1}  \n- z2={net.z2}  \n- yhat={net.yhat:.6f}\n\n"
            )
            file.write(f"**Pérdida BCE**: `{loss:.6f}`\n\n")
            file.write("**Gradientes**\n\n")
            file.write(
                f"- dW2={net.dW2}  \n- db2={net.db2}  \n- dW1={net.dW1}  \n- db1={net.db1}\n\n"
            )

    def log_update(self, net: MLP221) -> None:
        """Registra el estado de los pesos tras aplicar descenso de gradiente."""

        with open(self.path, "a", encoding="utf-8") as file:
            file.write("**Pesos después del update**\n\n")
            file.write(
                f"- W1={net.W1}  \n- b1={net.b1}  \n- W2={net.W2}  \n- b2={net.b2}\n\n"
            )

    def log_final_predictions(
        self, preds: Iterable[Tuple[Tuple[Sequence[float], float], float]]
    ) -> None:
        """Añade una tabla con las predicciones finales del modelo entrenado."""

        with open(self.path, "a", encoding="utf-8") as file:
            file.write("\n---\n\n## Predicciones finales\n\n")
            file.write("| x1 | x2 | y | ŷ |\n|---:|---:|---:|---:|\n")
            for (x, y), yhat in preds:
                file.write(
                    f"| {int(x[0])} | {int(x[1])} | {int(y)} | {yhat:.4f} |\n"
                )
            file.write("\n")
