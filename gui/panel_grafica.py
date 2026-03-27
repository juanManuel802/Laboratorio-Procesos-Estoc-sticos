"""
gui/panel_grafica.py

Widget reutilizable que encapsula un bloque gráfico completo:
un PlotWidget con una curva de referencia (offline, translúcida)
y una curva en vivo (online, sólida).

Por qué existe este módulo:
    Si main_window construyera los plots directamente, añadir una tercera
    gráfica requeriría duplicar manualmente toda la lógica de construcción
    y actualización. Al encapsularlo aquí, agregar un nuevo gráfico es
    solo instanciar otro PanelGrafica con distintos parámetros.

Responsabilidad única:
    Construir y actualizar UN bloque visual. No sabe de micrófono, de
    archivos de audio ni del clasificador.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt


class PanelGrafica(QWidget):
    """
    Un bloque visual autocontenido: título + PlotWidget con dos capas:

      - Curva de referencia (offline, translúcida): dibujada UNA vez al crear
        el panel, a partir de datos pregrabados del entrenamiento.
        Es opcional: si no se pasan datos de referencia, simplemente no aparece.

      - Curva en vivo (online, sólida): actualizable en cualquier momento
        mediante el método `actualizar()`.

    Parámetros
    ----------
    titulo      : str   Texto del título del plot.
    color_vivo  : str   Color hex de la curva en vivo. Ej: "#42a5f5"
    label_x     : str   Etiqueta del eje X.
    label_y     : str   Etiqueta del eje Y.
    freqs_ref   : np.ndarray | None  Eje X de la curva de referencia.
    datos_ref   : np.ndarray | None  Eje Y de la curva de referencia.
    color_ref   : tuple | None       Color RGBA de la curva de referencia.
                                     El canal A (alpha) controla la translucidez.
    nombre_ref  : str   Nombre en la leyenda de la curva de referencia.
    """

    def __init__(
        self,
        titulo: str,
        color_vivo: str,
        label_x: str = "",
        label_y: str = "",
        freqs_ref: np.ndarray | None = None,
        datos_ref: np.ndarray | None = None,
        color_ref: tuple | None = None,
        nombre_ref: str = "Referencia",
    ):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── PlotWidget ────────────────────────────────────────────────────────
        self._plot = pg.PlotWidget(title=titulo)
        self._plot.setBackground("#1c2d40")
        self._plot.showGrid(x=True, y=True, alpha=0.2)

        if label_x:
            self._plot.setLabel("bottom", label_x)
        if label_y:
            self._plot.setLabel("left", label_y)

        # ── Curva de referencia (offline, dibujada una sola vez) ──────────────
        #Por qué dibujamos la referencia ANTES que la curva viva:
        #pyqtgraph apila las curvas en el orden en que se agregan. Al poner la
        #referencia primero, queda visualmente "detrás" de la curva activa.
        tiene_referencia = (
            freqs_ref is not None
            and datos_ref is not None
            and color_ref is not None
        )
        if tiene_referencia:
            self._plot.addLegend()
            self._plot.plot(
                freqs_ref,
                datos_ref,
                pen=pg.mkPen(color=color_ref, width=1.5),
                name=nombre_ref,
            )

        # ── Curva en vivo (online, actualizable) ──────────────────────────────
        legend_kwargs = {"name": "Audio actual"} if tiene_referencia else {}
        self._curva_viva = self._plot.plot(
            pen=pg.mkPen(color=color_vivo, width=2),
            **legend_kwargs,
        )

        layout.addWidget(self._plot)

    # ── API pública ───────────────────────────────────────────────────────────

    def actualizar(self, eje_x: np.ndarray, eje_y: np.ndarray) -> None:
        """
        Reemplaza los datos de la curva en vivo.

        Parámetros
        ----------
        eje_x : np.ndarray  Valores del eje horizontal (tiempo o frecuencias).
        eje_y : np.ndarray  Valores del eje vertical (amplitud o magnitud).
        """
        self._curva_viva.setData(eje_x, eje_y)
