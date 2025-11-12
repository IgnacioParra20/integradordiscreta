import argparse
import tkinter as tk
from core import MLP221
from ui.app import App
from io.tracer import MarkdownTracer
from trainer.train import train
from io.export import export_loss_plot, export_pred_table

# opcional: benchmark numpy si existe
def maybe_numpy_bench(do_numpy: bool, epochs: int, lr: float):
    if not do_numpy:
        return
    try:
        from trainer.benchmark_numpy import numpy_benchmark
        numpy_benchmark(epochs=epochs, lr=lr)
    except Exception as e:
        print("Benchmark NumPy no disponible:", e)

def main():
    parser = argparse.ArgumentParser(description="Simulador MLP XOR 2–2–1 (BCE+Sigmoide) con grafo interactivo")
    parser.add_argument("--train", type=int, default=0, help="Entrenar N épocas antes de abrir la GUI")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate")
    parser.add_argument("--export", action="store_true", help="Exportar trazas/figuras/predicciones y salir")
    parser.add_argument("--numpy", action="store_true", help="Benchmark comparativo con NumPy (opcional)")
    args = parser.parse_args()

    net = MLP221()

    maybe_numpy_bench(args.numpy, max(args.train, 8000), args.lr)

    if args.export:
        tracer = MarkdownTracer("trazas.md")
        losses = train(net, epochs=max(args.train,3000), lr=args.lr, tracer=tracer)
        export_loss_plot(losses, "loss.png")
        export_pred_table(net, "predicciones.md")
        preds = [((x,y), net.predict(x)) for x,y in ([([0,0],0),([0,1],1),([1,0],1),([1,1],0)])]
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
