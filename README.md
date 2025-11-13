# Simulador MLP XOR 2–2–1

Este proyecto implementa un perceptrón multicapa (MLP) de arquitectura 2–2–1 para resolver el problema XOR. Incluye una interfaz gráfica didáctica construida con Tkinter que permite visualizar los pesos, activaciones y salidas del modelo en tiempo real, así como generar reportes en formato Markdown y gráficos de la curva de pérdida.

## ✨ Características principales

- 🎨 **Visualización interactiva**: Grafo animado que muestra la red neuronal con colores que indican activaciones y pesos
- 📊 **Entrenamiento en tiempo real**: Barra de progreso y métricas actualizadas durante el entrenamiento
- 📈 **Exportación de resultados**: Genera trazas detalladas, gráficos de pérdida y tablas de predicciones
- 🎓 **Enfoque didáctico**: Código explícito sin dependencias de NumPy para máxima claridad educativa
- ⚡ **Interfaz responsive**: Entrenamiento en segundo plano para mantener la UI fluida

## 📁 Contenido del repositorio

\`\`\`
├── core/              # Implementación del MLP y funciones auxiliares
│   ├── model.py       # Clase `MLP221` con forward, backward y step
│   ├── activations.py # Funciones de activación (sigmoide)
│   └── losses.py      # Función de pérdida BCE
├── data/
│   └── xor.py         # Conjunto de entrenamiento XOR
├── trainer/
│   └── train.py       # Bucles de entrenamiento (estándar y con callback)
├── mlpio/
│   ├── tracer.py      # Generación de bitácoras Markdown
│   └── export.py      # Exportación de gráfica de pérdida y tabla de predicciones
├── ui/
│   └── app.py         # Aplicación Tkinter interactiva
├── run.py             # Punto de entrada para lanzar la interfaz
└── requirements.txt   # Dependencias de Python
\`\`\`

## 🚀 Inicio rápido

### Requisitos previos

1. **Python 3.9+** (probado con Python 3.10 y 3.11)
2. Instalar las dependencias:

   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

   > **Nota**: `matplotlib` se usa únicamente para exportar la curva de pérdida (`loss.png`). La interfaz Tkinter forma parte de la biblioteca estándar de Python.

### Ejecutar el simulador

Desde la raíz del repositorio:

\`\`\`bash
python run.py
\`\`\`

Se abrirá una ventana titulada **"MLP XOR 2–2–1 — Simulador Didáctico"**.

### Opciones de línea de comandos

\`\`\`bash
# Entrenar 5000 épocas antes de abrir la GUI
python run.py --train 5000

# Ajustar learning rate
python run.py --train 3000 --lr 0.3

# Exportar sin abrir GUI
python run.py --export --train 3000

# Benchmark con NumPy (opcional)
python run.py --numpy
\`\`\`

## 🎮 Guía de uso de la interfaz

### Panel de control

#### Botones de entrada XOR
- **x=00, x=01, x=10, x=11**: Ejecuta un forward pass con la entrada correspondiente
- El botón seleccionado se destaca en naranja
- La visualización actualiza:
  - Neuronas de entrada (amarillo para 1, gris para 0)
  - Neuronas ocultas (intensidad azul según activación)
  - Neurona de salida (gradiente rojo según predicción)
  - Pesos visualizados con color y grosor proporcional

#### Controles de entrenamiento

- **LR (Learning Rate)**: Tasa de aprendizaje (recomendado: 0.3 - 0.7)
- **Épocas**: Número de iteraciones de entrenamiento completo

#### Botones de acción

1. **Entrenar**
   - Inicia el entrenamiento con los parámetros especificados
   - Muestra barra de progreso en tiempo real
   - Actualiza métricas: época actual, pérdida y precisión
   - Al finalizar guarda automáticamente:
     - `trazas.md`: Registro detallado paso a paso
     - `loss.png`: Gráfico de la curva de pérdida
     - `predicciones.md`: Tabla con predicciones finales

2. **Exportar trazas/figuras**
   - Genera los archivos de reporte sin entrenar
   - Útil para documentar el estado actual del modelo

3. **Reiniciar pesos**
   - Restablece los pesos a sus valores iniciales
   - Limpia el historial de entrenamiento
   - Reinicia la visualización

### Panel de información

Muestra en tiempo real:
- **x**: Vector de entrada actual
- **y**: Etiqueta objetivo
- **z1**: Pre-activaciones de la capa oculta
- **a1**: Activaciones de la capa oculta (post-sigmoid)
- **z2**: Pre-activación de la salida
- **ŷ**: Predicción del modelo
- **Precisión**: Porcentaje de aciertos en el dataset XOR

### Visualización del grafo

#### Codificación de colores

**Neuronas:**
- 🟡 **Amarillo**: Neuronas de entrada (más intenso cuando valor = 1)
- 🔵 **Azul**: Neuronas ocultas (intensidad según activación)
- 🔴 **Rojo**: Neurona de salida (gradiente según predicción)

**Conexiones (pesos):**
- 🔵 **Azul**: Pesos positivos (más oscuro = mayor magnitud)
- 🔴 **Rojo**: Pesos negativos (más oscuro = mayor magnitud)
- **Grosor**: Proporcional al valor absoluto del peso

## 🧠 Arquitectura del modelo

### Especificaciones técnicas

\`\`\`
Entrada (2) → Oculta (2) → Salida (1)
             sigmoid      sigmoid
\`\`\`

- **Función de activación**: Sigmoid en todas las capas
- **Función de pérdida**: Binary Cross Entropy (BCE)
- **Optimizador**: Gradient Descent (implementación manual)
- **Inicialización**: Pesos fijos para reproducibilidad

### Pesos iniciales

\`\`\`python
W1 = [[ 4.0,  4.0],   # h1 ← [x1, x2]
      [-4.0, -4.0]]   # h2 ← [x1, x2]
b1 = [-2.0, 6.0]

W2 = [[6.0, 6.0]]     # y ← [h1, h2]
b2 = [-9.0]
\`\`\`

Estos valores están preajustados cerca de una solución del problema XOR para facilitar el aprendizaje.

## 📊 Archivos generados

### trazas.md
Registro detallado del entrenamiento incluyendo:
- Pesos y sesgos antes de cada actualización
- Activaciones y logits de cada capa
- Gradientes calculados en backward
- Pesos actualizados después de cada paso
- Tabla de predicciones finales

### loss.png
Gráfico matplotlib que muestra:
- Eje X: Épocas
- Eje Y: Pérdida promedio (BCE)
- Permite visualizar la convergencia del modelo

### predicciones.md
Tabla markdown con:
- Entradas XOR (x1, x2)
- Etiquetas verdaderas (y)
- Predicciones del modelo (ŷ)

## 🎯 Flujo de trabajo recomendado

1. **Exploración inicial**
   \`\`\`bash
   python run.py
   \`\`\`
   - Prueba los 4 botones de entrada
   - Observa las activaciones con pesos iniciales

2. **Primer entrenamiento**
   - Configura LR=0.5, Épocas=3000
   - Presiona "Entrenar"
   - Observa cómo cambian colores y grosores

3. **Análisis de resultados**
   - Revisa `loss.png` para ver la convergencia
   - Examina `trazas.md` para entender el proceso
   - Verifica `predicciones.md` para la precisión final

4. **Experimentación**
   - Prueba diferentes learning rates (0.1, 0.5, 1.0)
   - Varía el número de épocas
   - Usa "Reiniciar pesos" para comparar experimentos

## 🛠️ Solución de problemas

### Tkinter no abre en Linux
\`\`\`bash
sudo apt-get install python3-tk
\`\`\`

### Entrenamiento lento
- Reduce el número de épocas para pruebas rápidas
- La actualización visual cada 10% del progreso mantiene la UI responsive

### Los pesos no convergen
- Aumenta el número de épocas (prueba 5000-10000)
- Ajusta el learning rate (prueba valores entre 0.3 y 0.7)
- Verifica que los pesos iniciales no estén muy alejados

### Limpiar archivos generados
\`\`\`bash
rm trazas.md loss.png predicciones.md
\`\`\`

## 🧪 Modo exportación (sin GUI)

Para generar reportes directamente:

\`\`\`bash
python run.py --export --train 3000 --lr 0.5
\`\`\`

Útil para:
- Integración en pipelines automatizados
- Generación de reportes batch
- Servidores sin display

## 🔬 Detalles de implementación

### ¿Por qué listas en vez de NumPy?

Este proyecto usa listas de Python y bucles explícitos para:
- **Claridad educativa**: Cada operación es explícita y fácil de seguir
- **Entendimiento profundo**: Los estudiantes ven exactamente qué hace cada línea
- **Sin abstracciones**: No hay "magia" detrás de operaciones vectorizadas

### Estabilidad numérica

- **Sigmoid**: Implementación dual para evitar overflow
- **BCE**: Clipping con epsilon para prevenir log(0)
- **Gradientes**: Uso de derivada simplificada para BCE+Sigmoid

### Threading para UI responsive

El entrenamiento se ejecuta en un thread separado:
- Callback periódico actualiza la UI
- Progreso visible en tiempo real
- Botones deshabilitados durante entrenamiento
- Manejo robusto de errores

## 📚 Recursos adicionales

### Para estudiantes

- El código está extensamente comentado en español
- Cada función tiene docstrings explicativos
- Los nombres de variables son descriptivos
- La estructura del proyecto es modular y fácil de navegar

### Para instructores

- Ideal para clases de Machine Learning introductorio
- Los estudiantes pueden modificar fácilmente:
  - Funciones de activación (`core/activations.py`)
  - Funciones de pérdida (`core/losses.py`)
  - Arquitectura de la red (`core/model.py`)
- Las trazas detalladas facilitan debugging conceptual

## 🤝 Contribuciones

Este es un proyecto educativo. Las contribuciones son bienvenidas, especialmente:
- Mejoras en la documentación
- Nuevas visualizaciones
- Funciones de activación/pérdida adicionales
- Tests automatizados
- Traducciones a otros idiomas

## 📄 Licencia

Este proyecto está diseñado con fines educativos. Siéntete libre de usar, modificar y distribuir el código.

---

**¡Disfruta explorando cómo aprende un MLP a resolver el XOR con una interfaz visual e interactiva!** 🎉
