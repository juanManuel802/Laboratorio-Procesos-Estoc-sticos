import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton
)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

from audio.captura import CapturaAudio
from classifier.clasificador import predecir_senal
from signal_processing.analisis import calcular_fft, calcular_magnitud

BUFFER = 2048
SR     = 44100


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

        #Título
        titulo = QLabel("Laboratorio Reconocimiento de Ruido Blanco")
        titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        titulo.setStyleSheet("font-size: 20px; font-weight: bold; color: #90caf9;")
        layout.addWidget(titulo)

        #Gráfica señal de audio
        self.graf_onda = pg.PlotWidget(title="Señal de audio")
        self.graf_onda.setBackground("#1c2d40")
        self.graf_onda.showGrid(x=True, y=True, alpha=0.2)
        self.graf_onda.setYRange(-1, 1)
        self.curva_onda = self.graf_onda.plot(
            np.zeros(BUFFER),
            pen=pg.mkPen(color="#42a5f5", width=1.5)
        )
        layout.addWidget(self.graf_onda)

        #Gráficadel espectro
        self.graf_psd = pg.PlotWidget(title="Espectro de potencia (PSD)")
        self.graf_psd.setBackground("#1c2d40")
        self.graf_psd.showGrid(x=True, y=True, alpha=0.2)
        self.graf_psd.setLabel("bottom", "Frecuencia (Hz)")
        self.graf_psd.setLabel("left", "dB")
        self.curva_psd = self.graf_psd.plot(
            pen=pg.mkPen(color="#ce93d8", width=1.5)
        )
        layout.addWidget(self.graf_psd)

        # UI: Resultado de clasificación
        self.lbl_resultado = QLabel("Presiona iniciar para grabar (2s)")
        self.lbl_resultado.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_resultado.setStyleSheet("font-size: 22px; font-weight: bold; color: #a5d6a7;")
        layout.addWidget(self.lbl_resultado)

        #Botones
        fila = QHBoxLayout()
        fila.setSpacing(12)

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

        self.btn_iniciar.clicked.connect(self.iniciar)

        fila.addStretch()
        fila.addWidget(self.btn_iniciar)
        fila.addStretch()
        layout.addLayout(fila)

        # ── Motor de Captura ──
        self.captura = CapturaAudio(callback=self.procesar_audio_grabado)

    def iniciar(self):
        self.btn_iniciar.setEnabled(False)
        self.lbl_resultado.setText("Grabando... (2s)")
        self.lbl_resultado.setStyleSheet("font-size: 24px; font-weight: bold; color: #ef5350;")
        self.captura.iniciar()

    def procesar_audio_grabado(self, senal, error=None):
        self.btn_iniciar.setEnabled(True)
        if error is not None:
            self.lbl_resultado.setText(f"Error de micrófono: {error}")
            return
            
        # 1. Mostrar la onda real en el tiempo
        # senal tiene 88200 puntos, graficaremos un submuestreo para agilidad visual
        paso = max(1, len(senal) // BUFFER)
        onda_reducida = senal[::paso]
        self.curva_onda.setData(onda_reducida)

        # 2. Calcular y mostrar espectro
        fft_vals = calcular_fft(senal)
        freqs, psd = calcular_magnitud(fft_vals, SR)
        self.curva_psd.setData(freqs, psd)

        # 3. Clasificar matemáticamente
        try:
            clase = predecir_senal(senal)
            if clase == "RUIDO BLANCO":
                color = "#4fc3f7"  # Azul claro
            else:
                color = "#ffca28"  # Amarillo
            self.lbl_resultado.setText(f"¡Resultado: {clase}!")
            self.lbl_resultado.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {color};")
        except Exception as e:
            self.lbl_resultado.setText("Error: Asegúrate de haber entrenado el modelo.")
            print(f"Error interno: {e}")


if __name__ == "__main__":
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec())