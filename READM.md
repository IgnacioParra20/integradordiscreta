# Simulador MLP XOR 2–2–1

Este proyecto implementa un perceptrón multicapa (MLP) de arquitectura 2–2–1 para resolver el problema XOR. Incluye una interfaz gráfica didáctica construida con Tkinter que permite visualizar los pesos, activaciones y salidas del modelo en tiempo real, así como generar reportes en formato Markdown y gráficos de la curva de pérdida.

## Contenido del repositorio

```
├── core/              # Implementación del MLP y funciones auxiliares
│   ├── model.py       # Clase `MLP221` con forward, backward y step
│   ├── activations.py # Funciones de activación (sigmoide)
│   └── losses.py      # Función de pérdida BCE
├── data/
│   └── xor.py         # Conjunto de entrenamiento XOR
├── trainer/
│   └── train.py       # Bucle de entrenamiento con opción de traza
├── mlpio/
│   ├── tracer.py      # Generación de bitácoras Markdown
│   └── export.py      # Exportación de gráfica de pérdida y tabla de predicciones
├── ui/
│   └── app.py         # Aplicación Tkinter interactiva
├── run.py             # Punto de entrada para lanzar la interfaz
└── requirements.txt   # Dependencias de Python
```

## Requisitos previos

1. **Python 3.9+** (se ha probado con Python 3.10).
2. Instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

   > `matplotlib` se usa únicamente para exportar la curva de pérdida (`loss.png`). La interfaz Tkinter forma parte de la biblioteca estándar de Python.

## Cómo ejecutar la interfaz

1. Desde la raíz del repositorio, lanza la aplicación:

   ```bash
   python run.py
   ```

2. Se abrirá una ventana titulada **"MLP XOR 2–2–1 — Simulador Didáctico"** con el siguiente diseño principal:
   - Un **lienzo** donde se dibuja la red (entradas, neuronas ocultas y salida) y se actualizan los pesos y sesgos.
   - Controles para ajustar la **tasa de aprendizaje (LR)** y el número de **épocas** antes de entrenar.
   - Botones de acción para ejecutar entradas del XOR, entrenar, exportar reportes o reiniciar los pesos.
   - Un panel inferior con datos numéricos del último forward (activaciones, logits y predicción).

## Recorrido por la interfaz

### Botones de ejemplo (`x=00`, `x=01`, `x=10`, `x=11`)

- Al pulsar cualquiera de estos botones, el modelo realiza un forward con la entrada correspondiente.
- La interfaz marca el botón seleccionado en color naranja y actualiza el panel inferior con:
  - Entrada (`x`), etiqueta (`y`), logits (`z1`, `z2`), activaciones ocultas (`a1`) y predicción final (`ŷ`).
- En el grafo:
  - Las neuronas de entrada, oculta y salida cambian de color y grosor para reflejar la activación.
  - Las aristas muestran colores azules (pesos positivos) o rojizos (pesos negativos) con intensidad proporcional al valor absoluto.

### Entrenar

- El botón **Entrenar** realiza entrenamiento batch sobre el dataset XOR usando los parámetros indicados (épocas y LR).
- Durante el entrenamiento se genera una lista de pérdidas por época y se actualizan los pesos del modelo.
- Al finalizar aparece un mensaje informando que se guardaron tres archivos en el directorio actual:
  - `trazas.md`: registro detallado del entrenamiento (pesos antes/después, gradientes, pérdidas por muestra).
  - `loss.png`: gráfico de la curva de pérdida promedio por época.
  - `predicciones.md`: tabla Markdown con las predicciones finales del modelo.
- La visualización del grafo se actualiza para reflejar los nuevos pesos y sesgos.

### Exportar trazas/figuras

- Útil si deseas generar los reportes sin volver a entrenar.
- Si no hay historial de entrenamiento cargado en memoria (`self.losses`), se crea una traza mínima recorriendo una época manual para documentar estados intermedios.
- Siempre reescribe `predicciones.md` y, si existen pérdidas recientes, también actualiza `loss.png`.

### Reiniciar pesos

- Restablece los pesos y sesgos iniciales definidos en `core/model.py`.
- Limpia los indicadores visuales y devuelve la interfaz a su estado original.

## Estructura del modelo MLP

El MLP implementado (`core/model.py`) tiene:
- **2 neuronas de entrada** (`x1`, `x2`).
- **2 neuronas ocultas** con activación sigmoide.
- **1 neurona de salida** sigmoide para clasificar XOR.
- Pérdida binaria (`BCE`).
- Métodos `forward`, `backward` y `step` explícitos para fines pedagógicos.

El entrenamiento (`trainer/train.py`) recorre las 4 combinaciones de XOR en cada época, acumula la pérdida promedio y permite registrar cada paso a través de `MarkdownTracer`.

## Generación de reportes

- `mlpio/tracer.py` controla la bitácora Markdown (`trazas.md`) con los estados del modelo antes y después de cada actualización.
- `mlpio/export.py` ofrece utilidades para:
  - Guardar la curva de pérdida (`export_loss_plot`).
  - Generar una tabla Markdown con las predicciones actuales (`export_pred_table`).

Ambas funciones se invocan desde la interfaz mediante los botones **Entrenar** y **Exportar trazas/figuras**.

## Ejemplo de flujo sugerido

1. Ejecuta `python run.py` para abrir la interfaz.
2. Prueba los cuatro botones de entrada para observar las activaciones iniciales (pesos preentrenados).
3. Ajusta LR y épocas si lo deseas y presiona **Entrenar**.
4. Observa cómo cambian colores y grosores de las conexiones según los nuevos pesos.
5. Genera reportes con **Exportar trazas/figuras** y revisa los archivos `trazas.md`, `loss.png` y `predicciones.md` en el directorio del proyecto.
6. Usa **Reiniciar pesos** para volver al estado inicial y repetir el experimento.

## Solución de problemas

- Si Tkinter no abre la ventana en Linux, instala los paquetes del sistema correspondientes (`sudo apt-get install python3-tk`).
- Para regenerar un entorno limpio, puedes borrar los archivos generados (`trazas.md`, `loss.png`, `predicciones.md`) y ejecutar nuevamente los pasos anteriores.

---

¡Disfruta explorando cómo aprende un MLP a resolver el XOR con una interfaz visual e interactiva!