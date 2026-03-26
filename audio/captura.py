"""
audio/captura.py

Módulo de captura de audio en tiempo real desde el micrófono.

Única responsabilidad: grabar exactamente 2 segundos de audio
y devolver un array numpy listo para ser procesado.

NO hace matemáticas, NO toca la GUI, NO toma decisiones.
Se ejecuta en un hilo separado para no bloquear la interfaz gráfica.
"""

import threading
import numpy as np
import sounddevice as sd


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES — deben coincidir con el resto del sistema
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE = 44100
DURACION_SEG = 2
N_MUESTRAS = SAMPLE_RATE * DURACION_SEG  # 88200 muestras


# ─────────────────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class CapturaAudio:
    """
    Graba audio desde el micrófono en un hilo separado.

    La GUI crea una instancia, llama a iniciar() y cuando la grabación
    termina se ejecuta automáticamente el callback que ella misma proveyó.

    Ejemplo de uso desde la GUI:
        def al_terminar(senal):
            resultado = clasificador.predecir(senal)
            self.mostrar_resultado(resultado)

        self.captura = CapturaAudio(callback=al_terminar)
        self.captura.iniciar()
    """

    def __init__(self, callback):
        """
        Parámetros
        ----------
        callback : callable
            Función que se llama al terminar la grabación.
            Recibe un argumento: array numpy de forma (88200,) float32.
            Es la GUI quien define qué hacer con ese array.
        """
        self._callback = callback
        self._hilo = None
        self._grabando = False

    # ── API pública ───────────────────────────────────────────────────────────

    def iniciar(self):
        """
        Arranca la grabación en un hilo separado.
        Retorna inmediatamente — no bloquea la GUI.
        """
        if self._grabando:
            return  # ya hay una grabación en curso, ignorar

        self._grabando = True
        self._hilo = threading.Thread(target=self._grabar, daemon=True)
        self._hilo.start()

    def esta_grabando(self) -> bool:
        """Indica si hay una grabación activa en este momento."""
        return self._grabando

    # ── Lógica interna ────────────────────────────────────────────────────────

    def _grabar(self):
        """
        Corre en el hilo secundario. Graba N_MUESTRAS del micrófono
        y llama al callback con el array resultante.

        sounddevice.rec() es bloqueante dentro del hilo — espera los
        2 segundos completos antes de continuar. Pero como corre en
        un hilo separado, la GUI no se congela.
        """
        try:
            # Grabar: N_MUESTRAS a SAMPLE_RATE Hz, 1 canal (mono), float32
            # sd.rec() inicia la grabación y sd.wait() espera a que termine
            senal_stereo = sd.rec(
                frames=N_MUESTRAS,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()  # bloquea este hilo hasta completar los 2 segundos

            # sd.rec devuelve shape (N_MUESTRAS, 1) → aplanar a (N_MUESTRAS,)
            senal = senal_stereo.flatten()

            # Entregar el array a quien esté escuchando (la GUI)
            self._callback(senal)

        except Exception as e:
            # Si el micrófono no está disponible u ocurre otro error,
            # notificar a la GUI con None para que pueda mostrar el error
            self._callback(None, error=str(e))

        finally:
            self._grabando = False