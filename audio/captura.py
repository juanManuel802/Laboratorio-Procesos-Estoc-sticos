import threading
import numpy as np
import sounddevice as sd

SAMPLE_RATE  = 44100
DURACION_SEG = 2
N_MUESTRAS   = SAMPLE_RATE * DURACION_SEG

class CapturaAudio:

    def __init__(self, callback):
        self._callback = callback
        self._hilo     = None
        self._grabando = False

    def iniciar(self):
        if self._grabando:
            return
        self._grabando = True
        self._hilo = threading.Thread(target=self._grabar, daemon=True)
        self._hilo.start()

    def esta_grabando(self) -> bool:
        return self._grabando

    def _grabar(self):
        try:
            senal_stereo = sd.rec(
                frames=N_MUESTRAS,
                samplerate=SAMPLE_RATE,
                channels=2,
                dtype="float32",
            )
            sd.wait()

            if senal_stereo.ndim == 2:
                senal = senal_stereo[:, 0]
            else:
                senal = senal_stereo.flatten()

        except Exception as e:
            self._grabando = False
            self._callback(None, error=str(e))
            return

        self._grabando = False
        self._callback(senal)