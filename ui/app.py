import tkinter as tk
from tkinter import ttk, messagebox
from core import MLP221
from data.xor import DATA
from mlpio.tracer import MarkdownTracer
from mlpio.export import export_loss_plot, export_pred_table
from trainer.train import train

class App:
    def __init__(self, root, net: MLP221):
        self.root = root
        self.root.title("MLP XOR 2–2–1 — Simulador Didáctico (BCE + Sigmoide)")
        self.net = net
        self.lr = 0.5
        self.epochs = 3000
        self.losses = []
        self._build_ui()
        self._draw_graph()
        self._update_labels([0.0,0.0], 0.0)

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(top, width=760, height=420, bg="#111")
        self.canvas.grid(row=0, column=0, columnspan=4, sticky="nsew", pady=(0,8))
        top.columnconfigure((0,1,2,3), weight=1)
        top.rowconfigure(0, weight=1)

        self.var_lr = tk.DoubleVar(value=self.lr)
        self.var_ep = tk.IntVar(value=self.epochs)

        ttk.Label(top, text="LR:").grid(row=1, column=0, sticky="e")
        ttk.Entry(top, textvariable=self.var_lr, width=6).grid(row=1, column=1, sticky="w")
        ttk.Label(top, text="Épocas:").grid(row=1, column=2, sticky="e")
        ttk.Entry(top, textvariable=self.var_ep, width=8).grid(row=1, column=3, sticky="w")

        btns = ttk.Frame(top)
        btns.grid(row=2, column=0, columnspan=4, pady=6)

        ttk.Button(btns, text="x=00", command=lambda:self.run_one([0.0,0.0],0.0)).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="x=01", command=lambda:self.run_one([0.0,1.0],1.0)).grid(row=0, column=1, padx=4)
        ttk.Button(btns, text="x=10", command=lambda:self.run_one([1.0,0.0],1.0)).grid(row=0, column=2, padx=4)
        ttk.Button(btns, text="x=11", command=lambda:self.run_one([1.0,1.0],0.0)).grid(row=0, column=3, padx=4)

        ttk.Button(btns, text="Entrenar", command=self.train_click).grid(row=0, column=4, padx=8)
        ttk.Button(btns, text="Exportar trazas/figuras", command=self.export_click).grid(row=0, column=5, padx=8)
        ttk.Button(btns, text="Reiniciar pesos", command=self.reset_weights).grid(row=0, column=6, padx=8)

        self.lbl_info = ttk.Label(top, text="", justify="left")
        self.lbl_info.grid(row=3, column=0, columnspan=4, sticky="w")

    def _draw_graph(self):
        self.canvas.delete("all")
        self.pos = {
            "x1": (100, 120), "x2": (100, 300),
            "h1": (350, 80),  "h2": (350, 340),
            "y":  (600, 210)
        }
        self.nodes = {}
        for name, (cx,cy) in self.pos.items():
            r = 28
            fill = "#ffd54a" if name.startswith("x") else ("#4fc3f7" if name.startswith("h") else "#ff8a80")
            self.nodes[name] = self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill=fill, outline="#eee", width=2)
            self.canvas.create_text(cx, cy, text=name, fill="#111")

        self.edges = []
        self.edge_labels = []

        def draw_edge(a,b, text):
            (x1,y1) = self.pos[a]
            (x2,y2) = self.pos[b]
            e = self.canvas.create_line(x1+30, y1, x2-30, y2, arrow=tk.LAST, fill="#aaa", width=2)
            self.edges.append(e)
            tx = (x1+x2)/2
            ty = (y1+y2)/2 - 14
            t = self.canvas.create_text(tx, ty, text=text, fill="#9ef")
            self.edge_labels.append(t)

        draw_edge("x1","h1", f"{self.net.W1[0][0]:+.2f}")
        draw_edge("x2","h1", f"{self.net.W1[0][1]:+.2f}")
        draw_edge("x1","h2", f"{self.net.W1[1][0]:+.2f}")
        draw_edge("x2","h2", f"{self.net.W1[1][1]:+.2f}")
        draw_edge("h1","y",  f"{self.net.W2[0][0]:+.2f}")
        draw_edge("h2","y",  f"{self.net.W2[0][1]:+.2f}")

        self.lbl_b1 = self.canvas.create_text(350, 30, text=f"b1={self.net.b1}", fill="#cfd8dc")
        self.lbl_b2 = self.canvas.create_text(600, 30, text=f"b2={self.net.b2}", fill="#cfd8dc")

    def _refresh_weight_labels(self):
        texts = [
            f"{self.net.W1[0][0]:+.2f}", f"{self.net.W1[0][1]:+.2f}",
            f"{self.net.W1[1][0]:+.2f}", f"{self.net.W1[1][1]:+.2f}",
            f"{self.net.W2[0][0]:+.2f}", f"{self.net.W2[0][1]:+.2f}",
        ]
        for i,t in enumerate(texts):
            self.canvas.itemconfigure(self.edge_labels[i], text=t)
        self.canvas.itemconfigure(self.lbl_b1, text=f"b1={ [round(v,2) for v in self.net.b1] }")
        self.canvas.itemconfigure(self.lbl_b2, text=f"b2={ [round(v,2) for v in self.net.b2] }")

    def _update_labels(self, x, y):
        info = [
            f"x={x}  y={int(y)}",
            f"z1={ [round(v,4) for v in self.net.z1] }",
            f"a1={ [round(v,4) for v in self.net.a1] }",
            f"z2={ [round(v,4) for v in self.net.z2] }",
            f"ŷ ={ self.net.yhat:.6f}",
        ]
        self.lbl_info.config(text=" | ".join(info))

    def run_one(self, x, y):
        yhat = self.net.forward(x)
        self._update_labels(x,y)
        shade = int(255*(1.0 - yhat))
        col = f"#{255:02x}{shade:02x}{shade:02x}"
        self.canvas.itemconfig(self.nodes["y"], fill=col)

    def train_click(self):
        try:
            self.lr = float(self.var_lr.get())
            self.epochs = int(self.var_ep.get())
        except:
            messagebox.showerror("Error", "LR o Épocas inválidos")
            return
        tracer = MarkdownTracer("trazas.md")
        self.losses = train(self.net, epochs=self.epochs, lr=self.lr, tracer=tracer)
        export_loss_plot(self.losses, "loss.png")
        export_pred_table(self.net, "predicciones.md")
        self._refresh_weight_labels()
        messagebox.showinfo("Listo", "Entrenamiento finalizado.\nSe guardaron: trazas.md, loss.png y predicciones.md")

    def export_click(self):
        if not self.losses:
            tracer = MarkdownTracer("trazas.md")
            tracer.log_epoch_header(0, self.lr)
            for x,y in DATA:
                self.net.forward(x)
                self.net.backward(y)
                tracer.log_sample(x,y,self.net)
                tracer.log_update(self.net)
            tracer.log_final_predictions([((x,y), self.net.predict(x)) for x,y in DATA])
        else:
            export_loss_plot(self.losses, "loss.png")
        export_pred_table(self.net, "predicciones.md")
        messagebox.showinfo("Exportado", "Se guardaron trazas.md, loss.png y predicciones.md")

    def reset_weights(self):
        self.__init__(self.root, MLP221())
        self._draw_graph()
