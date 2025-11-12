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
        self.option_buttons = {}
        self.edge_order = []
        self.edge_items = {}
        self.edge_labels = []
        self._configure_style()
        self._build_ui()
        self._draw_graph()
        self._update_labels([0.0,0.0], 0.0)

    def _configure_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#141829")
        style.configure("TLabel", background="#141829", foreground="#e8edf9")
        style.configure("TButton", padding=6, font=("Segoe UI", 10))
        style.configure("Accent.TButton", padding=8, font=("Segoe UI", 10, "bold"),
                        foreground="#ffffff", background="#5c6bc0")
        style.map(
            "Accent.TButton",
            background=[
                ("pressed", "#3949ab"),
                ("active", "#7986cb"),
                ("selected", "#ff7043"),
            ],
            relief=[("pressed", "sunken"), ("!pressed", "raised")],
        )
        style.configure("Option.TButton", padding=6, font=("Segoe UI", 10))
        style.map(
            "Option.TButton",
            background=[
                ("selected", "#ff7043"),
                ("pressed", "#ef6c00"),
                ("active", "#ffb74d"),
            ],
            foreground=[("selected", "#fff"), ("!selected", "#212121")],
        )

    def _build_ui(self):
        self.root.configure(bg="#0f1322")
        top = ttk.Frame(self.root, padding=12)
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

        for idx, (text, vector, target) in enumerate([
            ("x=00", [0.0, 0.0], 0.0),
            ("x=01", [0.0, 1.0], 1.0),
            ("x=10", [1.0, 0.0], 1.0),
            ("x=11", [1.0, 1.0], 0.0),
        ]):
            btn = ttk.Button(
                btns,
                text=text,
                style="Option.TButton",
                command=lambda v=vector, t=target: self.run_one(v, t),
            )
            btn.grid(row=0, column=idx, padx=4)
            self.option_buttons[tuple(vector)] = btn

        ttk.Button(btns, text="Entrenar", style="Accent.TButton", command=self.train_click).grid(row=0, column=4, padx=12)
        ttk.Button(btns, text="Exportar trazas/figuras", style="Accent.TButton", command=self.export_click).grid(row=0, column=5, padx=12)
        ttk.Button(btns, text="Reiniciar pesos", style="Accent.TButton", command=self.reset_weights).grid(row=0, column=6, padx=12)

        self.lbl_info = ttk.Label(top, text="", justify="left", font=("Consolas", 10))
        self.lbl_info.grid(row=3, column=0, columnspan=4, sticky="w")

    def _draw_graph(self):
        self.canvas.delete("all")
        self.pos = {
            "x1": (100, 120), "x2": (100, 300),
            "h1": (350, 80),  "h2": (350, 340),
            "y":  (600, 210)
        }
        self.nodes = {}
        self.base_node_colors = {
            "x": "#ffd54a",
            "h": "#4fc3f7",
            "y": "#ff8a80",
        }
        self.edge_labels = []
        self.edge_items = {}
        self.edge_order = []
        for name, (cx,cy) in self.pos.items():
            r = 28
            key = name[0]
            fill = self.base_node_colors.get(key, "#4fc3f7")
            self.nodes[name] = self.canvas.create_oval(
                cx-r,
                cy-r,
                cx+r,
                cy+r,
                fill=fill,
                outline="#e8edf9",
                width=2,
            )
            self.canvas.create_text(cx, cy, text=name, fill="#0f1322", font=("Segoe UI", 11, "bold"))

        def draw_edge(a,b, text):
            (x1,y1) = self.pos[a]
            (x2,y2) = self.pos[b]
            e = self.canvas.create_line(x1+30, y1, x2-30, y2, arrow=tk.LAST, fill="#90a4ae", width=2)
            self.edge_items[(a,b)] = e
            tx = (x1+x2)/2
            ty = (y1+y2)/2 - 14
            t = self.canvas.create_text(tx, ty, text=text, fill="#9ef", font=("Consolas", 10))
            self.edge_labels.append(t)
            self.edge_order.append((a,b))

        draw_edge("x1","h1", f"{self.net.W1[0][0]:+.2f}")
        draw_edge("x2","h1", f"{self.net.W1[0][1]:+.2f}")
        draw_edge("x1","h2", f"{self.net.W1[1][0]:+.2f}")
        draw_edge("x2","h2", f"{self.net.W1[1][1]:+.2f}")
        draw_edge("h1","y",  f"{self.net.W2[0][0]:+.2f}")
        draw_edge("h2","y",  f"{self.net.W2[0][1]:+.2f}")

        self.lbl_b1 = self.canvas.create_text(350, 30, text=f"b1={self.net.b1}", fill="#cfd8dc", font=("Consolas", 10))
        self.lbl_b2 = self.canvas.create_text(600, 30, text=f"b2={self.net.b2}", fill="#cfd8dc", font=("Consolas", 10))
        self._refresh_weight_labels()

    def _edge_color(self, weight):
        weight = max(min(weight, 2.0), -2.0)
        intensity = min(abs(weight) / 2.0, 1.0)
        if weight >= 0:
            start = (144, 202, 249)
            end = (30, 136, 229)
        else:
            start = (255, 171, 145)
            end = (229, 57, 53)
        r = int(start[0] + (end[0] - start[0]) * intensity)
        g = int(start[1] + (end[1] - start[1]) * intensity)
        b = int(start[2] + (end[2] - start[2]) * intensity)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _blend(self, start_hex, end_hex, t):
        t = max(0.0, min(1.0, t))
        sr, sg, sb = tuple(int(start_hex[i:i+2], 16) for i in (1, 3, 5))
        er, eg, eb = tuple(int(end_hex[i:i+2], 16) for i in (1, 3, 5))
        r = int(sr + (er - sr) * t)
        g = int(sg + (eg - sg) * t)
        b = int(sb + (eb - sb) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _reset_node_styles(self):
        for name, node in self.nodes.items():
            base = self.base_node_colors.get(name[0], "#4fc3f7")
            self.canvas.itemconfigure(node, fill=base, outline="#e8edf9", width=2)

    def _refresh_weight_labels(self):
        texts = [
            f"{self.net.W1[0][0]:+.2f}", f"{self.net.W1[0][1]:+.2f}",
            f"{self.net.W1[1][0]:+.2f}", f"{self.net.W1[1][1]:+.2f}",
            f"{self.net.W2[0][0]:+.2f}", f"{self.net.W2[0][1]:+.2f}",
        ]
        weights = [
            self.net.W1[0][0], self.net.W1[0][1],
            self.net.W1[1][0], self.net.W1[1][1],
            self.net.W2[0][0], self.net.W2[0][1],
        ]
        for i, t in enumerate(texts):
            color = self._edge_color(weights[i])
            self.canvas.itemconfigure(self.edge_labels[i], text=t, fill=color)
            edge_key = self.edge_order[i]
            self.canvas.itemconfigure(
                self.edge_items[edge_key],
                fill=color,
                width=2 + min(abs(weights[i]) * 1.5, 4),
            )
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
        self._reset_node_styles()
        for btn in self.option_buttons.values():
            btn.state(["!selected"])
        sel_btn = self.option_buttons.get(tuple(x))
        if sel_btn:
            sel_btn.state(["selected"])
        for idx, name in enumerate(["x1", "x2"]):
            if x[idx] >= 0.5:
                self.canvas.itemconfigure(
                    self.nodes[name],
                    fill=self._blend("#ffd54a", "#ff9800", 0.9),
                    outline="#ffcc80",
                    width=3,
                )
            else:
                self.canvas.itemconfigure(
                    self.nodes[name],
                    fill=self._blend("#37474f", "#ffd54a", 0.2),
                    outline="#90a4ae",
                    width=2,
                )

        hidden_vals = getattr(self.net, "a1", [0.0, 0.0])
        for idx, name in enumerate(["h1", "h2"]):
            val = hidden_vals[idx] if idx < len(hidden_vals) else 0.0
            color = self._blend("#263238", "#4fc3f7", val)
            outline = self._blend("#455a64", "#81d4fa", val)
            self.canvas.itemconfigure(
                self.nodes[name],
                fill=color,
                outline=outline,
                width=2 + val,
            )

        for i, edge_key in enumerate(self.edge_order):
            weight = (
                self.net.W1[i//2][i%2]
                if i < 4 else self.net.W2[0][i-4]
            )
            self.canvas.itemconfigure(
                self.edge_items[edge_key],
                fill=self._edge_color(weight),
                width=2 + min(abs(weight) * 1.5, 4),
            )

        shade = int(255*(1.0 - yhat))
        col = f"#{255:02x}{shade:02x}{shade:02x}"
        self.canvas.itemconfig(
            self.nodes["y"],
            fill=col,
            outline="#ffe082" if yhat > 0.5 else "#ffcdd2",
            width=3,
        )

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
