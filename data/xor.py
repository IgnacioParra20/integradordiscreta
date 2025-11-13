"""Conjunto de datos exhaustivo para el problema XOR clásico.

Cada muestra está representada como ``([x1, x2], y)`` con entradas binarias y la
etiqueta que la MLP debe aprender. Al estar declarado como lista simple resulta
fácil de iterar y de mostrar en tablas o trazas educativas.
"""

from typing import List, Tuple

# Lista de todas las combinaciones posibles de bits y su salida XOR.
DATA: List[Tuple[list, float]] = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]
