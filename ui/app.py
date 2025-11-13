import tkinter as tk
from tkinter import ttk, messagebox
from threading import Thread
from typing import Optional

from core import MLP221
from data.xor import DATA
from mlpio.tracer import MarkdownTracer
from mlpio.export import export_loss_plot, export_pred_table
from trainer.train import train_with_callback

class App:
    def __init__(self, root, net: MLP221):
        self.root = root
        self.root.title("MLP XOR 2-2-1 - Simulador Didáctico Interactivo")
        self.root.resizable(True, True)
        self.net = net
        self.lr = 0.5
        self.epochs = 3000
        self.losses = []
        self.is_training = False
        self.training_thread: Optional[Thread] = None
        
        self.option_buttons = {}
        self.edge_order = []
        self.edge_items = {}
        self.edge_labels = []
        
        self.root.minsize(900, 700)
        
        self._configure_style()
        self._build_ui()
        self._draw_graph()
        self._update_labels([0.0,0.0], 0.0)
        
        self.root.after(500, self._show_welcome_if_first_time)

    def _configure_style(self):
        """Configure modern visual theme with improved color palette."""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        
        self.colors = {
            "bg_dark": "#1a1d2e",
            "bg_medium": "#252a41",
            "bg_light": "#2d3354",
            "accent_primary": "#667eea",
            "accent_secondary": "#764ba2",
            "accent_success": "#48bb78",
            "accent_warning": "#f6ad55",
            "accent_danger": "#f56565",
            "text_primary": "#f7fafc",
            "text_secondary": "#cbd5e0",
            "text_muted": "#a0aec0",
        }
        
        style.configure("TFrame", background=self.colors["bg_dark"])
        style.configure("TLabel", 
                       background=self.colors["bg_dark"], 
                       foreground=self.colors["text_primary"],
                       font=("Segoe UI", 10))
        style.configure("Title.TLabel", 
                       background=self.colors["bg_dark"], 
                       foreground=self.colors["text_primary"],
                       font=("Segoe UI", 14, "bold"))
        style.configure("Subtitle.TLabel", 
                       background=self.colors["bg_dark"], 
                       foreground=self.colors["text_secondary"],
                       font=("Segoe UI", 11))
        
        style.configure("TButton", 
                       padding=(12, 8), 
                       font=("Segoe UI", 10),
                       borderwidth=0)
        
        style.configure("Primary.TButton", 
                       padding=(14, 10), 
                       font=("Segoe UI", 10, "bold"),
                       foreground="#ffffff", 
                       background=self.colors["accent_primary"])
        style.map("Primary.TButton",
                 background=[("pressed", "#5a67d8"), ("active", "#7c8aed"), ("disabled", "#4a5568")],
                 relief=[("pressed", "sunken"), ("!pressed", "raised")])
        
        style.configure("Success.TButton", 
                       padding=(14, 10), 
                       font=("Segoe UI", 10, "bold"),
                       foreground="#ffffff", 
                       background=self.colors["accent_success"])
        style.map("Success.TButton",
                 background=[("pressed", "#38a169"), ("active", "#68d391"), ("disabled", "#4a5568")])
        
        style.configure("Option.TButton", 
                       padding=(10, 8), 
                       font=("Consolas", 10, "bold"))
        style.map("Option.TButton",
                 background=[("selected", self.colors["accent_warning"]), 
                           ("pressed", "#ed8936"), 
                           ("active", "#fbd38d")],
                 foreground=[("selected", "#fff"), ("!selected", self.colors["text_primary"])])
        
        style.configure("Help.TButton",
                       padding=(10, 8),
                       font=("Segoe UI", 10),
                       foreground=self.colors["text_primary"])
        
        style.configure("Training.Horizontal.TProgressbar", 
                       background=self.colors["accent_success"], 
                       troughcolor=self.colors["bg_medium"],
                       bordercolor=self.colors["bg_dark"],
                       lightcolor="#68d391",
                       darkcolor="#38a169",
                       thickness=20)
        
        style.configure("TEntry",
                       fieldbackground=self.colors["bg_light"],
                       foreground=self.colors["text_primary"],
                       borderwidth=2,
                       relief="flat")

    def _build_ui(self):
        """Build the main user interface with improved layout."""
        self.root.configure(bg=self.colors["bg_dark"])
        
        main_container = ttk.Frame(self.root, padding=20)
        main_container.pack(fill="both", expand=True)
        
        header = ttk.Frame(main_container)
        header.pack(fill="x", pady=(0, 16))
        
        title_frame = ttk.Frame(header)
        title_frame.pack(side="left")
        
        title = ttk.Label(title_frame, 
                         text="Simulador de Red Neuronal XOR", 
                         style="Title.TLabel")
        title.pack(anchor="w")
        
        subtitle = ttk.Label(title_frame, 
                            text="MLP 2-2-1 con BCE y Sigmoide", 
                            style="Subtitle.TLabel")
        subtitle.pack(anchor="w")
        
        btn_help = ttk.Button(header, 
                             text="? Ayuda", 
                             style="Help.TButton",
                             command=self.show_help)
        btn_help.pack(side="right", padx=5)
        
        btn_tutorial = ttk.Button(header, 
                                 text="Tutorial", 
                                 style="Help.TButton",
                                 command=self.show_tutorial)
        btn_tutorial.pack(side="right", padx=5)
        
        canvas_frame = ttk.Frame(main_container)
        canvas_frame.pack(fill="both", expand=True, pady=(0, 16))
        
        self.canvas = tk.Canvas(canvas_frame, 
                               width=800, 
                               height=480, 
                               bg=self.colors["bg_medium"],
                               highlightthickness=2,
                               highlightbackground=self.colors["bg_light"])
        self.canvas.pack(fill="both", expand=True)
        
        controls = ttk.Frame(main_container)
        controls.pack(fill="x", pady=(0, 12))
        
        params_frame = ttk.Frame(controls)
        params_frame.pack(fill="x", pady=(0, 12))
        
        params_label = ttk.Label(params_frame, 
                                text="Parámetros de Entrenamiento:", 
                                font=("Segoe UI", 10, "bold"),
                                foreground=self.colors["text_secondary"])
        params_label.pack(anchor="w", pady=(0, 8))
        
        params_inputs = ttk.Frame(params_frame)
        params_inputs.pack(fill="x")
        
        self.var_lr = tk.DoubleVar(value=self.lr)
        self.var_ep = tk.IntVar(value=self.epochs)
        
        lr_container = ttk.Frame(params_inputs)
        lr_container.pack(side="left", padx=(0, 20))
        
        ttk.Label(lr_container, text="Learning Rate (LR):").pack(side="left", padx=(0, 8))
        lr_entry = ttk.Entry(lr_container, textvariable=self.var_lr, width=10)
        lr_entry.pack(side="left")
        self._create_tooltip(lr_entry, "Tasa de aprendizaje (típicamente 0.1 - 1.0)")
        
        ep_container = ttk.Frame(params_inputs)
        ep_container.pack(side="left")
        
        ttk.Label(ep_container, text="Épocas:").pack(side="left", padx=(0, 8))
        ep_entry = ttk.Entry(ep_container, textvariable=self.var_ep, width=10)
        ep_entry.pack(side="left")
        self._create_tooltip(ep_entry, "Número de iteraciones de entrenamiento")
        
        test_frame = ttk.Frame(controls)
        test_frame.pack(fill="x", pady=(0, 12))
        
        test_label = ttk.Label(test_frame, 
                              text="Probar Predicción:", 
                              font=("Segoe UI", 10, "bold"),
                              foreground=self.colors["text_secondary"])
        test_label.pack(anchor="w", pady=(0, 8))
        
        test_buttons = ttk.Frame(test_frame)
        test_buttons.pack(fill="x")
        
        for idx, (text, vector, target, tooltip) in enumerate([
            ("x = [0, 0]", [0.0, 0.0], 0.0, "Entrada: 0 XOR 0 = 0"),
            ("x = [0, 1]", [0.0, 1.0], 1.0, "Entrada: 0 XOR 1 = 1"),
            ("x = [1, 0]", [1.0, 0.0], 1.0, "Entrada: 1 XOR 0 = 1"),
            ("x = [1, 1]", [1.0, 1.0], 0.0, "Entrada: 1 XOR 1 = 0"),
        ]):
            btn = ttk.Button(
                test_buttons,
                text=text,
                style="Option.TButton",
                command=lambda v=vector, t=target: self.run_one(v, t),
            )
            btn.pack(side="left", padx=5)
            self.option_buttons[tuple(vector)] = btn
            self._create_tooltip(btn, tooltip)
        
        actions = ttk.Frame(controls)
        actions.pack(fill="x", pady=(0, 12))
        
        actions_label = ttk.Label(actions, 
                                 text="Acciones:", 
                                 font=("Segoe UI", 10, "bold"),
                                 foreground=self.colors["text_secondary"])
        actions_label.pack(anchor="w", pady=(0, 8))
        
        buttons_row = ttk.Frame(actions)
        buttons_row.pack(fill="x")
        
        self.btn_train = ttk.Button(buttons_row, 
                                    text="Entrenar Red", 
                                    style="Primary.TButton", 
                                    command=self.train_click)
        self.btn_train.pack(side="left", padx=(0, 10))
        self._create_tooltip(self.btn_train, "Entrenar la red con los parámetros configurados")
        
        self.btn_export = ttk.Button(buttons_row, 
                                     text="Exportar Datos", 
                                     style="Success.TButton", 
                                     command=self.export_click)
        self.btn_export.pack(side="left", padx=(0, 10))
        self._create_tooltip(self.btn_export, "Guardar trazas, gráficas y predicciones")
        
        self.btn_reset = ttk.Button(buttons_row, 
                                    text="Reiniciar Pesos", 
                                    style="TButton", 
                                    command=self.reset_weights)
        self.btn_reset.pack(side="left")
        self._create_tooltip(self.btn_reset, "Volver a los pesos iniciales aleatorios")
        
        progress_frame = ttk.Frame(main_container)
        progress_frame.pack(fill="x", pady=(0, 8))
        
        self.progress = ttk.Progressbar(
            progress_frame, 
            orient="horizontal", 
            mode="determinate",
            style="Training.Horizontal.TProgressbar"
        )
        self.progress.pack(fill="x")
        self.progress["maximum"] = 100
        self.progress["value"] = 0
        
        status_frame = ttk.Frame(main_container)
        status_frame.pack(fill="x")
        
        self.lbl_info = ttk.Label(status_frame, 
                                 text="", 
                                 justify="left", 
                                 font=("Consolas", 9),
                                 foreground=self.colors["text_secondary"])
        self.lbl_info.pack(anchor="w", pady=(0, 4))
        
        self.lbl_training = ttk.Label(
            status_frame, 
            text="Listo para entrenar", 
            justify="left", 
            font=("Segoe UI", 10),
            foreground=self.colors["accent_success"]
        )
        self.lbl_training.pack(anchor="w")

    def _create_tooltip(self, widget, text):
        """Create a tooltip for a widget."""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = tk.Label(tooltip, 
                           text=text, 
                           background=self.colors["bg_light"],
                           foreground=self.colors["text_primary"],
                           relief="solid",
                           borderwidth=1,
                           font=("Segoe UI", 9),
                           padx=8,
                           pady=4)
            label.pack()
            
            widget.tooltip = tooltip
        
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                delattr(widget, 'tooltip')
        
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def _show_welcome_if_first_time(self):
        """Show welcome dialog on first launch."""
        response = messagebox.askyesno(
            "Bienvenido al Simulador MLP XOR",
            "¿Es tu primera vez usando este simulador?\n\n"
            "¿Te gustaría ver un tutorial rápido?",
            icon='question'
        )
        if response:
            self.show_tutorial()

    def show_help(self):
        """Display help window with comprehensive usage information."""
        help_window = tk.Toplevel(self.root)
        help_window.title("Ayuda - Simulador MLP XOR")
        help_window.geometry("700x600")
        help_window.configure(bg=self.colors["bg_dark"])
        help_window.resizable(False, False)
        
        canvas = tk.Canvas(help_window, bg=self.colors["bg_dark"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(help_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        help_content = [
            ("¿Qué es este simulador?", 
             "Este es un simulador educativo de una Red Neuronal Multicapa (MLP) "
             "diseñada para resolver el problema XOR. La red tiene arquitectura 2-2-1 "
             "(2 entradas, 2 neuronas ocultas, 1 salida) y usa función de activación "
             "sigmoide con pérdida Binary Cross-Entropy (BCE)."),
            
            ("¿Qué es el problema XOR?",
             "XOR (OR Exclusivo) es una función lógica que devuelve 1 solo cuando "
             "las entradas son diferentes:\n"
             "• 0 XOR 0 = 0\n"
             "• 0 XOR 1 = 1\n"
             "• 1 XOR 0 = 1\n"
             "• 1 XOR 1 = 0\n\n"
             "Este problema no es linealmente separable, por lo que requiere una "
             "red neuronal con capa oculta."),
            
            ("Componentes de la Interfaz",
             "• Gráfico de Red: Visualización de la arquitectura con nodos y conexiones\n"
             "• Parámetros: Learning Rate y Épocas para controlar el entrenamiento\n"
             "• Botones de Prueba: Probar predicciones con las 4 combinaciones XOR\n"
             "• Botón Entrenar: Inicia el proceso de entrenamiento\n"
             "• Botón Exportar: Guarda trazas, gráficas y tablas de predicción\n"
             "• Botón Reiniciar: Vuelve a pesos iniciales aleatorios\n"
             "• Barra de Progreso: Muestra avance del entrenamiento\n"
             "• Panel de Estado: Información detallada de la red"),
            
            ("Cómo Usar el Simulador",
             "1. Configura los parámetros (LR y Épocas)\n"
             "2. Haz clic en 'Entrenar Red'\n"
             "3. Observa el progreso en tiempo real\n"
             "4. Prueba predicciones con los botones x=[0,0], etc.\n"
             "5. Exporta resultados si deseas guardar datos"),
            
            ("Interpretación del Gráfico",
             "• Nodos Amarillos: Neuronas de entrada (x1, x2)\n"
             "• Nodos Azules: Neuronas ocultas (h1, h2)\n"
             "• Nodo Rojo: Neurona de salida (y)\n"
             "• Flechas: Conexiones con pesos (números mostrados)\n"
             "• Color de Flechas: Indica magnitud y signo del peso\n"
             "  - Azul: Pesos positivos\n"
             "  - Rojo/Naranja: Pesos negativos\n"
             "  - Intensidad: Mayor peso = color más intenso"),
            
            ("Parámetros de Entrenamiento",
             "• Learning Rate (LR): Controla qué tan grandes son los ajustes "
             "de pesos en cada iteración. Valores típicos: 0.1 - 1.0\n"
             "  - Muy bajo: Aprendizaje lento\n"
             "  - Muy alto: Puede no converger\n\n"
             "• Épocas: Número de veces que la red ve todos los datos de "
             "entrenamiento. Para XOR, 3000-5000 épocas suelen ser suficientes."),
            
            ("Métricas de Evaluación",
             "• Pérdida (Loss): Mide qué tan equivocadas son las predicciones. "
             "Menor es mejor. BCE está en rango [0, ∞).\n\n"
             "• Precisión (Accuracy): Porcentaje de predicciones correctas. "
             "100% significa que la red resolvió perfectamente el XOR."),
            
            ("Archivos Exportados",
             "• trazas.md: Registro detallado paso a paso del entrenamiento\n"
             "• loss.png: Gráfica de la evolución de la pérdida\n"
             "• predicciones.md: Tabla con predicciones finales para cada entrada"),
        ]
        
        for i, (title, content) in enumerate(help_content):
            section = ttk.Frame(scrollable_frame, padding=15)
            section.pack(fill="x", padx=10, pady=5)
            
            title_label = ttk.Label(section, 
                                   text=title, 
                                   font=("Segoe UI", 11, "bold"),
                                   foreground=self.colors["accent_primary"])
            title_label.pack(anchor="w", pady=(0, 5))
            
            content_label = ttk.Label(section, 
                                     text=content, 
                                     font=("Segoe UI", 9),
                                     foreground=self.colors["text_secondary"],
                                     wraplength=620,
                                     justify="left")
            content_label.pack(anchor="w")
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        close_btn = ttk.Button(help_window, 
                              text="Cerrar", 
                              style="Primary.TButton",
                              command=help_window.destroy)
        close_btn.pack(pady=10)

    def show_tutorial(self):
        """Display interactive step-by-step tutorial."""
        tutorial_window = tk.Toplevel(self.root)
        tutorial_window.title("Tutorial Interactivo")
        tutorial_window.geometry("600x500")
        tutorial_window.configure(bg=self.colors["bg_dark"])
        tutorial_window.resizable(False, False)
        
        steps = [
            {
                "title": "Paso 1: Entender el Problema XOR",
                "content": "XOR es una función lógica que devuelve 1 solo cuando "
                          "las entradas son diferentes.\n\n"
                          "Problema: No es linealmente separable (no se puede resolver "
                          "con una línea recta).\n\n"
                          "Solución: Usar una red neuronal con capa oculta.",
                "image": None
            },
            {
                "title": "Paso 2: Arquitectura de la Red",
                "content": "Nuestra red MLP 2-2-1 tiene:\n\n"
                          "• 2 neuronas de entrada (x1, x2)\n"
                          "• 2 neuronas en capa oculta (h1, h2)\n"
                          "• 1 neurona de salida (y)\n\n"
                          "Total: 6 pesos + 3 sesgos = 9 parámetros a aprender",
                "image": None
            },
            {
                "title": "Paso 3: Configurar Parámetros",
                "content": "Antes de entrenar, configura:\n\n"
                          "• Learning Rate (LR): Recomendado 0.5\n"
                          "  Controla velocidad de aprendizaje\n\n"
                          "• Épocas: Recomendado 3000-5000\n"
                          "  Número de iteraciones de entrenamiento\n\n"
                          "Consejo: Empieza con valores por defecto.",
                "image": None
            },
            {
                "title": "Paso 4: Entrenar la Red",
                "content": "Haz clic en 'Entrenar Red':\n\n"
                          "• Observa la barra de progreso\n"
                          "• Mira cómo disminuye la pérdida\n"
                          "• La precisión debe llegar a 100%\n\n"
                          "Durante el entrenamiento, los pesos se ajustan "
                          "automáticamente para minimizar el error.",
                "image": None
            },
            {
                "title": "Paso 5: Probar Predicciones",
                "content": "Después de entrenar, usa los botones de prueba:\n\n"
                          "• x = [0, 0] → debe predecir ~0\n"
                          "• x = [0, 1] → debe predecir ~1\n"
                          "• x = [1, 0] → debe predecir ~1\n"
                          "• x = [1, 1] → debe predecir ~0\n\n"
                          "Observa cómo cambian los colores del gráfico.",
                "image": None
            },
            {
                "title": "Paso 6: Interpretar Resultados",
                "content": "Panel de estado muestra:\n\n"
                          "• x: Entrada actual\n"
                          "• z1, a1: Valores de capa oculta\n"
                          "• z2, ŷ: Salida de la red\n"
                          "• Precisión: % de aciertos\n\n"
                          "Colores del gráfico:\n"
                          "• Nodos más brillantes = más activados\n"
                          "• Flechas azules = pesos positivos\n"
                          "• Flechas rojas = pesos negativos",
                "image": None
            },
            {
                "title": "Paso 7: Exportar y Analizar",
                "content": "Usa 'Exportar Datos' para guardar:\n\n"
                          "• trazas.md: Log detallado del entrenamiento\n"
                          "• loss.png: Gráfica de evolución de pérdida\n"
                          "• predicciones.md: Tabla de resultados\n\n"
                          "Estos archivos son útiles para análisis profundo "
                          "y presentaciones.",
                "image": None
            },
            {
                "title": "Experimentación",
                "content": "Ahora experimenta:\n\n"
                          "1. Cambia el Learning Rate (prueba 0.1, 1.0, 2.0)\n"
                          "2. Usa diferentes cantidades de épocas\n"
                          "3. Observa cómo afecta la convergencia\n"
                          "4. Usa 'Reiniciar Pesos' para empezar de nuevo\n\n"
                          "Objetivo: Entender cómo los hiperparámetros "
                          "afectan el aprendizaje.",
                "image": None
            }
        ]
        
        current_step = [0]  # Use list for mutability in nested function
        
        content_frame = ttk.Frame(tutorial_window, padding=20)
        content_frame.pack(fill="both", expand=True)
        
        title_label = ttk.Label(content_frame, 
                               text="", 
                               font=("Segoe UI", 13, "bold"),
                               foreground=self.colors["accent_primary"],
                               wraplength=550)
        title_label.pack(anchor="w", pady=(0, 15))
        
        content_label = ttk.Label(content_frame, 
                                 text="", 
                                 font=("Segoe UI", 10),
                                 foreground=self.colors["text_secondary"],
                                 wraplength=550,
                                 justify="left")
        content_label.pack(anchor="w", fill="both", expand=True)
        
        progress_label = ttk.Label(content_frame,
                                  text="",
                                  font=("Segoe UI", 9),
                                  foreground=self.colors["text_muted"])
        progress_label.pack(pady=(15, 0))
        
        def update_step():
            step = steps[current_step[0]]
            title_label.config(text=step["title"])
            content_label.config(text=step["content"])
            progress_label.config(text=f"Paso {current_step[0] + 1} de {len(steps)}")
            
            btn_prev.config(state="normal" if current_step[0] > 0 else "disabled")
            btn_next.config(text="Siguiente" if current_step[0] < len(steps) - 1 else "Finalizar")
        
        def next_step():
            if current_step[0] < len(steps) - 1:
                current_step[0] += 1
                update_step()
            else:
                tutorial_window.destroy()
        
        def prev_step():
            if current_step[0] > 0:
                current_step[0] -= 1
                update_step()
        
        nav_frame = ttk.Frame(tutorial_window)
        nav_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        btn_prev = ttk.Button(nav_frame, 
                             text="Anterior", 
                             command=prev_step)
        btn_prev.pack(side="left")
        
        btn_skip = ttk.Button(nav_frame, 
                             text="Saltar Tutorial", 
                             command=tutorial_window.destroy)
        btn_skip.pack(side="left", padx=10)
        
        btn_next = ttk.Button(nav_frame, 
                             text="Siguiente", 
                             style="Primary.TButton",
                             command=next_step)
        btn_next.pack(side="right")
        
        update_step()

    def _draw_graph(self):
        """Draw the neural network graph with improved visual design."""
        self.canvas.delete("all")
        
        self.pos = {
            "x1": (120, 140), "x2": (120, 340),
            "h1": (400, 120),  "h2": (400, 360),
            "y":  (680, 240)
        }
        
        self.nodes = {}
        
        self.base_node_colors = {
            "x": "#fbbf24",  # Amber
            "h": "#60a5fa",  # Blue
            "y": "#f87171",  # Red
        }
        
        self.edge_labels = []
        self.edge_items = {}
        self.edge_order = []
        
        def draw_edge(a, b, text):
            (x1, y1) = self.pos[a]
            (x2, y2) = self.pos[b]
            e = self.canvas.create_line(x1+35, y1, x2-35, y2, 
                                       arrow=tk.LAST, 
                                       fill="#64748b", 
                                       width=3,
                                       smooth=True)
            self.edge_items[(a, b)] = e
            tx = (x1 + x2) / 2
            ty = (y1 + y2) / 2 - 18
            
            bg = self.canvas.create_rectangle(tx-25, ty-10, tx+25, ty+10,
                                             fill=self.colors["bg_light"],
                                             outline=self.colors["bg_medium"],
                                             width=2)
            t = self.canvas.create_text(tx, ty, 
                                       text=text, 
                                       fill=self.colors["text_primary"], 
                                       font=("Consolas", 10, "bold"))
            self.edge_labels.append(t)
            self.edge_order.append((a, b))
        
        draw_edge("x1", "h1", f"{self.net.W1[0][0]:+.2f}")
        draw_edge("x2", "h1", f"{self.net.W1[0][1]:+.2f}")
        draw_edge("x1", "h2", f"{self.net.W1[1][0]:+.2f}")
        draw_edge("x2", "h2", f"{self.net.W1[1][1]:+.2f}")
        draw_edge("h1", "y",  f"{self.net.W2[0][0]:+.2f}")
        draw_edge("h2", "y",  f"{self.net.W2[0][1]:+.2f}")
        
        for name, (cx, cy) in self.pos.items():
            r = 32
            key = name[0]
            fill = self.base_node_colors.get(key, "#60a5fa")
            
            # Shadow effect
            self.canvas.create_oval(cx-r+3, cy-r+3, cx+r+3, cy+r+3,
                                   fill="#1e293b", outline="")
            
            # Main node
            self.nodes[name] = self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                fill=fill,
                outline="#f1f5f9",
                width=3
            )
            self.canvas.create_text(cx, cy, 
                                   text=name, 
                                   fill="#1e293b", 
                                   font=("Segoe UI", 12, "bold"))
        
        self.lbl_b1 = self.canvas.create_text(400, 50, 
                                             text=f"b1={self.net.b1}", 
                                             fill=self.colors["text_secondary"], 
                                             font=("Consolas", 10))
        self.lbl_b2 = self.canvas.create_text(680, 50, 
                                             text=f"b2={self.net.b2}", 
                                             fill=self.colors["text_secondary"], 
                                             font=("Consolas", 10))
        
        self._refresh_weight_labels()

    def _edge_color(self, weight):
        """Calculate edge color based on weight magnitude and sign."""
        weight = max(min(weight, 3.0), -3.0)
        intensity = min(abs(weight) / 3.0, 1.0)
        if weight >= 0:
            start = (147, 197, 253)  # Light blue
            end = (37, 99, 235)      # Dark blue
        else:
            start = (254, 202, 202)  # Light red
            end = (239, 68, 68)      # Dark red
        r = int(start[0] + (end[0] - start[0]) * intensity)
        g = int(start[1] + (end[1] - start[1]) * intensity)
        b = int(start[2] + (end[2] - start[2]) * intensity)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _blend(self, start_hex, end_hex, t):
        """Blend two hex colors based on parameter t (0-1)."""
        t = max(0.0, min(1.0, t))
        sr, sg, sb = tuple(int(start_hex[i:i+2], 16) for i in (1, 3, 5))
        er, eg, eb = tuple(int(end_hex[i:i+2], 16) for i in (1, 3, 5))
        r = int(sr + (er - sr) * t)
        g = int(sg + (eg - sg) * t)
        b = int(sb + (eb - sb) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _reset_node_styles(self):
        """Reset all nodes to their base colors."""
        for name, node in self.nodes.items():
            base = self.base_node_colors.get(name[0], "#60a5fa")
            self.canvas.itemconfigure(node, fill=base, outline="#f1f5f9", width=3)

    def _refresh_weight_labels(self):
        """Update all weight and bias labels on the graph."""
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
            self.canvas.itemconfigure(self.edge_labels[i], text=t, fill=self.colors["text_primary"])
            edge_key = self.edge_order[i]
            self.canvas.itemconfigure(
                self.edge_items[edge_key],
                fill=color,
                width=3 + min(abs(weights[i]) * 1.2, 5),
            )
        self.canvas.itemconfigure(self.lbl_b1, text=f"b1={ [round(v,2) for v in self.net.b1] }")
        self.canvas.itemconfigure(self.lbl_b2, text=f"b2={ [round(v,2) for v in self.net.b2] }")

    def _update_labels(self, x, y):
        """Update the info panel with current network state."""
        accuracy = self._calculate_accuracy()
        info = [
            f"x={x}  y={int(y)}",
            f"z1={ [round(v,4) for v in self.net.z1] }",
            f"a1={ [round(v,4) for v in self.net.a1] }",
            f"z2={ [round(v,4) for v in self.net.z2] }",
            f"ŷ ={ self.net.yhat:.6f}",
            f"Precisión={accuracy:.1f}%"
        ]
        self.lbl_info.config(text=" | ".join(info))

    def _calculate_accuracy(self) -> float:
        """Calculate current model accuracy on XOR dataset."""
        correct = 0
        for x, y in DATA:
            pred = self.net.predict(x)
            if (pred > 0.5 and y == 1.0) or (pred <= 0.5 and y == 0.0):
                correct += 1
        return (correct / len(DATA)) * 100

    def run_one(self, x, y):
        """Execute forward pass for one input and update visualization."""
        yhat = self.net.forward(x)
        self._update_labels(x, y)
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
                    fill=self._blend("#fbbf24", "#f59e0b", 0.8),
                    outline="#fcd34d",
                    width=4,
                )
            else:
                self.canvas.itemconfigure(
                    self.nodes[name],
                    fill=self._blend("#475569", "#fbbf24", 0.3),
                    outline="#94a3b8",
                    width=2,
                )

        hidden_vals = getattr(self.net, "a1", [0.0, 0.0])
        for idx, name in enumerate(["h1", "h2"]):
            val = hidden_vals[idx] if idx < len(hidden_vals) else 0.0
            color = self._blend("#1e3a8a", "#60a5fa", val)
            outline = self._blend("#475569", "#93c5fd", val)
            self.canvas.itemconfigure(
                self.nodes[name],
                fill=color,
                outline=outline,
                width=2 + val * 2,
            )

        for i, edge_key in enumerate(self.edge_order):
            weight = (
                self.net.W1[i//2][i%2]
                if i < 4 else self.net.W2[0][i-4]
            )
            self.canvas.itemconfigure(
                self.edge_items[edge_key],
                fill=self._edge_color(weight),
                width=3 + min(abs(weight) * 1.2, 5),
            )

        shade = int(255 * (1.0 - yhat))
        col = f"#{255:02x}{shade:02x}{shade:02x}"
        self.canvas.itemconfig(
            self.nodes["y"],
            fill=col,
            outline="#fde047" if yhat > 0.5 else "#fca5a5",
            width=4,
        )

    def _training_callback(self, epoch: int, total_epochs: int, loss: float):
        """Callback executed during training to update UI."""
        progress = (epoch / total_epochs) * 100
        self.root.after(0, lambda: self._update_training_status(epoch, total_epochs, loss, progress))

    def _update_training_status(self, epoch: int, total_epochs: int, loss: float, progress: float):
        """Update UI elements during training (must run in main thread)."""
        self.progress["value"] = progress
        accuracy = self._calculate_accuracy()
        self.lbl_training.config(
            text=f"Época {epoch}/{total_epochs} | Pérdida: {loss:.6f} | Precisión: {accuracy:.1f}%",
            foreground=self.colors["accent_success"] if epoch < total_epochs else self.colors["accent_primary"]
        )
        if epoch % max(1, total_epochs // 10) == 0:
            self._refresh_weight_labels()
        self.root.update_idletasks()

    def _run_training_thread(self):
        """Execute training in a separate thread to keep UI responsive."""
        try:
            tracer = MarkdownTracer("trazas.md")
            self.losses = train_with_callback(
                self.net, 
                epochs=self.epochs, 
                lr=self.lr, 
                tracer=tracer,
                callback=self._training_callback
            )
            export_loss_plot(self.losses, "loss.png")
            export_pred_table(self.net, "predicciones.md")
            
            self.root.after(0, self._training_complete)
        except Exception as e:
            self.root.after(0, lambda: self._training_error(str(e)))

    def _training_complete(self):
        """Called when training finishes successfully."""
        self.is_training = False
        self._refresh_weight_labels()
        self._enable_controls()
        accuracy = self._calculate_accuracy()
        self.lbl_training.config(
            text=f"Entrenamiento completado | Precisión final: {accuracy:.1f}%",
            foreground=self.colors["accent_success"]
        )
        messagebox.showinfo(
            "Entrenamiento Completado", 
            f"El entrenamiento ha finalizado exitosamente.\n\n"
            f"Precisión: {accuracy:.1f}%\n"
            f"Pérdida final: {self.losses[-1] if self.losses else 0:.6f}\n\n"
            f"Archivos guardados:\n"
            f"  • trazas.md\n"
            f"  • loss.png\n"
            f"  • predicciones.md"
        )

    def _training_error(self, error_msg: str):
        """Called when training encounters an error."""
        self.is_training = False
        self._enable_controls()
        self.progress["value"] = 0
        self.lbl_training.config(
            text="Error durante entrenamiento",
            foreground=self.colors["accent_danger"]
        )
        messagebox.showerror("Error", f"Error durante el entrenamiento:\n{error_msg}")

    def _disable_controls(self):
        """Disable all control buttons during training."""
        self.btn_train.state(["disabled"])
        self.btn_export.state(["disabled"])
        self.btn_reset.state(["disabled"])
        for btn in self.option_buttons.values():
            btn.state(["disabled"])

    def _enable_controls(self):
        """Re-enable all control buttons after training."""
        self.btn_train.state(["!disabled"])
        self.btn_export.state(["!disabled"])
        self.btn_reset.state(["!disabled"])
        for btn in self.option_buttons.values():
            btn.state(["!disabled"])

    def train_click(self):
        """Handle training button click with validation and threading."""
        if self.is_training:
            messagebox.showwarning("Advertencia", "Ya hay un entrenamiento en progreso")
            return
            
        try:
            self.lr = float(self.var_lr.get())
            self.epochs = int(self.var_ep.get())
            
            if self.lr <= 0 or self.lr > 10:
                raise ValueError("Learning rate debe estar entre 0 y 10")
            if self.epochs <= 0 or self.epochs > 100000:
                raise ValueError("Épocas debe estar entre 1 y 100000")
                
        except ValueError as e:
            messagebox.showerror("Error de validación", str(e))
            return
        
        self.is_training = True
        self._disable_controls()
        self.progress["value"] = 0
        self.lbl_training.config(
            text="Iniciando entrenamiento...",
            foreground=self.colors["accent_warning"]
        )
        
        self.training_thread = Thread(target=self._run_training_thread, daemon=True)
        self.training_thread.start()

    def export_click(self):
        """Export traces and figures without training."""
        if not self.losses:
            tracer = MarkdownTracer("trazas.md")
            tracer.log_epoch_header(0, self.lr)
            for x, y in DATA:
                self.net.forward(x)
                self.net.backward(y)
                tracer.log_sample(x, y, self.net)
                tracer.log_update(self.net)
            tracer.log_final_predictions([((x, y), self.net.predict(x)) for x, y in DATA])
        else:
            export_loss_plot(self.losses, "loss.png")
        export_pred_table(self.net, "predicciones.md")
        messagebox.showinfo("Exportado", "Se guardaron:\n  • trazas.md\n  • loss.png\n  • predicciones.md")

    def reset_weights(self):
        """Reset network to initial weights and clear training history."""
        if self.is_training:
            messagebox.showwarning("Advertencia", "No se puede reiniciar durante el entrenamiento")
            return
            
        self.net = MLP221()
        self.losses = []
        self.progress["value"] = 0
        self.lbl_training.config(
            text="Pesos reiniciados - Listo para entrenar",
            foreground=self.colors["accent_success"]
        )
        self._draw_graph()
        self._update_labels([0.0, 0.0], 0.0)
        messagebox.showinfo("Reiniciado", "Los pesos se han reiniciado a sus valores iniciales")
