"""
gui/main_window.py

Ventana principal de la aplicación.

Responsabilidad única:
    Orquestar los componentes visuales y conectar el motor de captura
    con el clasificador. No contiene lógica matemática ni de hardware.

Escalabilidad:
    Añadir una nueva gráfica es solo instanciar otro PanelGrafica y
    llamar a su método `actualizar()` dentro del callback de captura.
"""

import os
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
)
from PyQt6.QtCore import Qt

from audio.captura import CapturaAudio
from classifier.clasificador import predecir_senal, CARPETA_PAT
from signal_processing.analisis import normalizar_rms, autocovarianza_discreta, calcular_fft, calcular_magnitud
from gui.panel_grafica import PanelGrafica


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES VISUALES
# ─────────────────────────────────────────────────────────────────────────────

BUFFER      = 2048    # puntos máximos a dibujar en la gráfica de onda
SAMPLE_RATE = 44100


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _cargar_patrones():
    
    ruta_rb   = os.path.join(CARPETA_PAT, "espectro_promedio_rb.npy")
    ruta_em   = os.path.join(CARPETA_PAT, "espectro_promedio_em.npy")

    if os.path.exists(ruta_rb) and os.path.exists(ruta_em):
        pat_rb = np.load(ruta_rb)
        pat_em = np.load(ruta_em)
        eje_x = np.arange(len(pat_rb))
        return eje_x, pat_rb, pat_em

    return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# VENTANA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class VentanaPrincipal(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detector de Ruido Blanco")
        self.setMinimumSize(900, 600)
        self.setStyleSheet("background-color: #0d1b2a; color: #e0e0e0;")

        central = QWidget()
        self.setCentralWidget(central)
        layout  = QVBoxLayout(central)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # ── Título ────────────────────────────────────────────────────────────
        titulo = QLabel("Laboratorio — Reconocimiento de Ruido Blanco")
        titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        titulo.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #90caf9;"
        )
        layout.addWidget(titulo)

        # ── Cargar patrones de entrenamiento (pueden no existir aún) ─────────
        eje_x, pat_rb, pat_em = _cargar_patrones()

        # ── Panel: onda en el tiempo ─────────────────────────────────────────
        # No tiene referencia de entrenamiento porque promediar fases no tiene
        # sentido físico: la media de N ondas arbitrarias tiende a cero.
        self.panel_onda = PanelGrafica(
            titulo="Señal de audio (tiempo)",
            color_vivo="#42a5f5",
            label_x="Muestras",
            label_y="Amplitud",
        )
        layout.addWidget(self.panel_onda)

        # ── Panel: espectro de potencia ──────────────────────────────────────
        # Aquí sí tiene sentido mostrar los patrones promedio del entrenamiento:
        # el espectro de potencia es estable entre grabaciones de la misma clase.
        # Se dibujan ambas referencias para que el usuario vea en cuál "cae" su audio.
        self.panel_psd = PanelGrafica(
            titulo="Espectros",
            color_vivo="#ce93d8",
            label_x="hola",
            label_y="Magnitud",
            freqs_ref=eje_x,
            datos_ref=pat_rb,
            color_ref=(79, 195, 247, 60),    # cian translúcido → Ruido Blanco
            nombre_ref="Ref: Ruido Blanco",
        )
        # Segunda referencia (Emisora): se agrega directamente al PlotWidget interno
        # usando la API pública de PanelGrafica._plot para no romper la encapsulación
        # del método actualizar(), que solo maneja la curva viva.
        if eje_x is not None and pat_em is not None:
            self.panel_psd._plot.plot(
                eje_x,
                pat_em,
                pen=pg.mkPen(color=(255, 202, 40, 60), width=1.5),
                name="Ref: Emisora",
            )
        layout.addWidget(self.panel_psd)

        # ── Label de resultado ───────────────────────────────────────────────
        self.lbl_resultado = QLabel("Presiona iniciar para grabar (2 s)")
        self.lbl_resultado.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_resultado.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #a5d6a7;"
        )
        layout.addWidget(self.lbl_resultado)

        # ── Botón ─────────────────────────────────────────────────────────────
        fila = QHBoxLayout()
        self.btn_iniciar = QPushButton("▶  Iniciar grabación")
        self.btn_iniciar.setFixedHeight(38)
        self.btn_iniciar.setStyleSheet("""
            QPushButton {
                background-color: #1565c0;
                color: white;
                border-radius: 6px;
                font-size: 13px;
                padding: 0 24px;
            }
            QPushButton:hover    { background-color: #1976d2; }
            QPushButton:disabled { background-color: #263238; color: #546e7a; }
        """)
        self.btn_iniciar.clicked.connect(self._iniciar)
        fila.addStretch()
        fila.addWidget(self.btn_iniciar)
        fila.addStretch()
        layout.addLayout(fila)

        # ── Motor de captura ──────────────────────────────────────────────────
        self._captura = CapturaAudio(callback=self._procesar_audio_grabado)

    # ── Slots / callbacks ─────────────────────────────────────────────────────

    def _iniciar(self):
        self.btn_iniciar.setEnabled(False)
        self.lbl_resultado.setText("Grabando... (2 s)")
        self.lbl_resultado.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #ef5350;"
        )
        self._captura.iniciar()

    def _procesar_audio_grabado(self, senal: np.ndarray, error: str | None = None):
        self.btn_iniciar.setEnabled(True)

        if error is not None:
            self.lbl_resultado.setText(f"Error de micrófono: {error}")
            return

        # 1. Onda: submuestreo adaptativo para no saturar el render
        paso = max(1, len(senal) // BUFFER)
        self.panel_onda.actualizar(
            np.arange(len(senal[::paso])),
            senal[::paso],
        )

        # 2. Espectro de potencia: mismo pipeline que entrenamiento
        #    normalizar → autocovarianza → FFT → magnitud
        senal_norm = normalizar_rms(senal)
        autocov_vals = autocovarianza_discreta(senal_norm)
        fft_vals = calcular_fft(autocov_vals)
        psd = calcular_magnitud(fft_vals, SAMPLE_RATE)
        eje_x = np.arange(len(psd))
        self.panel_psd.actualizar(psd)

        # 3. Clasificar y mostrar veredicto
        try:
            clase = predecir_senal(senal)
            color = "#4fc3f7" if clase == "RUIDO BLANCO" else "#ffca28"
            self.lbl_resultado.setText(f"¡Resultado: {clase}!")
            self.lbl_resultado.setStyleSheet(
                f"font-size: 28px; font-weight: bold; color: {color};"
            )
        except FileNotFoundError:
            self.lbl_resultado.setText(
                "Sin patrones — ejecuta primero: python main.py train"
            )


if __name__ == "__main__":
    pg.setConfigOptions(antialias=True)
    app = QApplication([])
    app.setStyle("Fusion")
    ventana = VentanaPrincipal()
    ventana.show()
    app.exit(app.exec())