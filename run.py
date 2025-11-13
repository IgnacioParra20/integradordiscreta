"""Punto de entrada del simulador interactivo de la MLP XOR 2–2–1.

El script ofrece una interfaz de línea de comandos que permite elegir entre
entrenar el modelo, exportar resultados detallados (trazas en Markdown, curva de
pérdida y tabla de predicciones) o abrir la interfaz gráfica de Tkinter.
"""

import argparse
import tkinter as tk

from core import MLP221
from mlpio.export import export_loss_plot, export_pred_table
from mlpio.tracer import MarkdownTracer
from trainer.train import train
from ui.app import App


def maybe_numpy_bench(do_numpy: bool, epochs: int, lr: float) -> None:
    """Ejecuta el benchmark vectorizado en NumPy si el usuario lo solicita.

    Args:
        do_numpy: ``True`` si debe ejecutarse el benchmark.
        epochs: Número de épocas para la comparación (se usa un valor alto
            para ilustrar diferencias de rendimiento).
        lr: Tasa de aprendizaje empleada durante el benchmark.
    """

    if not do_numpy:
        return
    try:
        from trainer.benchmark_numpy import numpy_benchmark

        numpy_benchmark(epochs=epochs, lr=lr)
    except Exception as exc:  # pragma: no cover - feedback puramente informativo
        print("Benchmark NumPy no disponible:", exc)


def main() -> None:
    """Analiza argumentos CLI y coordina entrenamiento, exportación o GUI."""

    parser = argparse.ArgumentParser(
        description="Simulador MLP XOR 2–2–1 (BCE+Sigmoide) con grafo interactivo"
    )
    parser.add_argument(
        "--train", type=int, default=0, help="Entrenar N épocas antes de abrir la GUI"
    )
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate")
    parser.add_argument(
        "--export",
        action="store_true",
        help="Exportar trazas/figuras/predicciones y salir",
    )
    parser.add_argument(
        "--numpy",
        action="store_true",
        help="Benchmark comparativo con NumPy (opcional)",
    )
    args = parser.parse_args()

    net = MLP221()

    if args.export:
        tracer = MarkdownTracer("trazas.md")
        losses = train(net, epochs=max(args.train, 3000), lr=args.lr, tracer=tracer)
        export_loss_plot(losses, "loss.png")
        export_pred_table(net, "predicciones.md")
        preds = [
            ((x, y), net.predict(x))
            for x, y in [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
        ]
        tracer.log_final_predictions(preds)
        print("Exportado: trazas.md, loss.png, predicciones.md")
        return

    if args.train > 0:
        tracer = MarkdownTracer("trazas.md")
        losses = train(net, epochs=args.train, lr=args.lr, tracer=tracer)
        export_loss_plot(losses, "loss.png")
        export_pred_table(net, "predicciones.md")

    root = tk.Tk()
    App(root, net)
    root.mainloop()


if __name__ == "__main__":
    main()
