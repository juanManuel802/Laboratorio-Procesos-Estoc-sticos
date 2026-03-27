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
from signal_processing.analisis import normalizar, calcular_espectro_directo
from gui.panel_grafica import PanelGrafica

BUFFER      = 2048
SAMPLE_RATE = 44100

def _cargar_patrones():
    ruta_rb = os.path.join(CARPETA_PAT, "espectro_promedio_rb.npy")
    ruta_em = os.path.join(CARPETA_PAT, "espectro_promedio_em.npy")
    if os.path.exists(ruta_rb) and os.path.exists(ruta_em):
        pat_rb = np.load(ruta_rb)
        pat_em = np.load(ruta_em)
        return np.arange(len(pat_rb)), pat_rb, pat_em
    return None, None, None


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

        titulo = QLabel("Laboratorio — Reconocimiento de Ruido Blanco")
        titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        titulo.setStyleSheet("font-size: 20px; font-weight: bold; color: #90caf9;")
        layout.addWidget(titulo)

        eje_x, pat_rb, pat_em = _cargar_patrones()

        self.panel_onda = PanelGrafica(
            titulo="Señal de audio (tiempo)",
            color_vivo="#42a5f5",
            label_x="Muestras",
            label_y="Amplitud (x0.001)",
        )
        layout.addWidget(self.panel_onda)

        self.panel_psd = PanelGrafica(
            titulo="Espectro (FFT)",
            color_vivo="#ce93d8",
            label_x="Frecuencia (Hz)",
            label_y="Magnitud",
            freqs_ref=eje_x,
            datos_ref=pat_rb,
            color_ref=(79, 195, 247, 80),
            nombre_ref="Ref: Ruido Blanco",
        )
        if eje_x is not None and pat_em is not None:
            self.panel_psd._plot.plot(
                eje_x, pat_em,
                pen=pg.mkPen(color=(255, 202, 40, 80), width=1.5),
                name="Ref: Emisora",
            )
        layout.addWidget(self.panel_psd)

        self.lbl_resultado = QLabel("Presiona iniciar para grabar (2 s)")
        self.lbl_resultado.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_resultado.setStyleSheet("font-size: 22px; font-weight: bold; color: #a5d6a7;")
        layout.addWidget(self.lbl_resultado)

        fila = QHBoxLayout()
        self.btn_iniciar = QPushButton("▶  Iniciar grabación")
        self.btn_iniciar.setFixedHeight(38)
        self.btn_iniciar.setStyleSheet("""
            QPushButton {
                background-color: #1565c0; color: white;
                border-radius: 6px; font-size: 13px; padding: 0 24px;
            }
            QPushButton:hover    { background-color: #1976d2; }
            QPushButton:disabled { background-color: #263238; color: #546e7a; }
        """)
        self.btn_iniciar.clicked.connect(self._iniciar)
        fila.addStretch()
        fila.addWidget(self.btn_iniciar)
        fila.addStretch()
        layout.addLayout(fila)

        self._captura = CapturaAudio(callback=self._procesar_audio_grabado)

    def _iniciar(self):
        self.btn_iniciar.setEnabled(False)
        self.lbl_resultado.setText("Grabando... (2 s)")
        self.lbl_resultado.setStyleSheet("font-size: 24px; font-weight: bold; color: #ef5350;")
        self._captura.iniciar()

    def _procesar_audio_grabado(self, senal: np.ndarray, error: str | None = None):
        self.btn_iniciar.setEnabled(True)

        if error is not None:
            self.lbl_resultado.setText(f"Error de micrófono: {error}")
            return

        # Gráfica de onda
        paso = max(1, len(senal) // BUFFER)
        self.panel_onda.actualizar(
            np.arange(len(senal[::paso])),
            senal[::paso] * 1000,   # escalar para visualización
        )

        # Gráfica de espectro — mismo pipeline que clasificador
        senal_norm = normalizar(senal)
        espectro   = calcular_espectro_directo(senal_norm)
        # Eje X en Hz
        n_bins = len(espectro)
        eje_hz = np.linspace(0, SAMPLE_RATE / 2, n_bins)
        self.panel_psd.actualizar(eje_hz, espectro)

        # Clasificar
        try:
            clase = predecir_senal(senal)
            color = "#4fc3f7" if clase == "RUIDO BLANCO" else "#ffca28"
            self.lbl_resultado.setText(f"¡Resultado: {clase}!")
            self.lbl_resultado.setStyleSheet(
                f"font-size: 28px; font-weight: bold; color: {color};"
            )
        except FileNotFoundError:
            self.lbl_resultado.setText("Sin patrones — ejecuta: python main.py train")


if __name__ == "__main__":
    pg.setConfigOptions(antialias=True)
    app = QApplication([])
    app.setStyle("Fusion")
    ventana = VentanaPrincipal()
    ventana.show()
    app.exit(app.exec())