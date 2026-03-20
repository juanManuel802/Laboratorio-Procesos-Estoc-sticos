import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton
)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg


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

        self.btn_detener = QPushButton("■  Detener")
        self.btn_detener.setFixedHeight(38)
        self.btn_detener.setEnabled(False)
        self.btn_detener.setStyleSheet("""
            QPushButton {
                background-color: #37474f;
                color: #b0bec5;
                border-radius: 6px;
                font-size: 13px;
                padding: 0 24px;
                border: 1px solid #455a64;
            }
            QPushButton:hover    { background-color: #455a64; color: white; }
            QPushButton:disabled { background-color: #1c2d40; color: #37474f; border-color: #263238; }
        """)

        self.btn_iniciar.clicked.connect(self.iniciar)
        self.btn_detener.clicked.connect(self.detener)

        fila.addStretch()
        fila.addWidget(self.btn_iniciar)
        fila.addWidget(self.btn_detener)
        fila.addStretch()
        layout.addLayout(fila)

        # ── Timer para datos demo ──
        self._timer = QTimer()
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._actualizar)

    def iniciar(self):
        # TODO: conectar con audio/captura.py
        self.btn_iniciar.setEnabled(False)
        self.btn_detener.setEnabled(True)
        self._timer.start()

    def detener(self):
        #detener captura boton
        self.btn_iniciar.setEnabled(True)
        self.btn_detener.setEnabled(False)
        self._timer.stop()

    def _actualizar(self):
        #Datos falsos pa probar
        onda  = np.random.normal(0, 0.2, BUFFER)
        freqs = np.linspace(0, SR // 2, 512)
        psd   = np.random.uniform(-60, -40, 512)

        self.curva_onda.setData(onda)
        self.curva_psd.setData(freqs, psd)


if __name__ == "__main__":
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec())