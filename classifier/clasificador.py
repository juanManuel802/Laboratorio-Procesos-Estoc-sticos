"""
classifier/clasificador.py

Lógica para predecir si un archivo de audio de entrada
es ruido blanco o de una emisora de radio.
"""

import os
import numpy as np
import librosa

from signal_processing.analisis import normalizar_rms, autocovarianza_discreta, calcular_fft, calcular_magnitud

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE    = 44100
DURACION_SEG   = 2
N_MUESTRAS     = SAMPLE_RATE * DURACION_SEG

CARPETA_PAT    = os.path.join(os.path.dirname(__file__), "..", "data", "patterns")
RUTA_PAT_RB    = os.path.join(CARPETA_PAT, "espectro_promedio_rb.npy")
RUTA_PAT_EM    = os.path.join(CARPETA_PAT, "espectro_promedio_em.npy")


def predecir_senal(senal: np.ndarray) -> str:
    """
    Recibe un array numpy de señal en crudo, calcula su espectro de magnitud
    y lo compara con los patrones entrenados usando distancia euclidiana.
    
    Parámetros
    ----------
    senal : np.ndarray
        Señal de audio mono a 44100 Hz.
        
    Retorna
    -------
    Clase predicha: "RUIDO BLANCO" o "EMISORA"
    """
    if not os.path.exists(RUTA_PAT_RB) or not os.path.exists(RUTA_PAT_EM):
        raise FileNotFoundError("No se encontraron los patrones. Corre el entrenamiento primero.")

    # 1. Asegurar el tamaño esperado
    if len(senal) > N_MUESTRAS:
        senal = senal[:N_MUESTRAS]
    elif len(senal) < N_MUESTRAS:
        senal = np.pad(senal, (0, N_MUESTRAS - len(senal)))

    # 2. Normalizar por RMS (igual que en entrenamiento)
    senal = normalizar_rms(senal)

    # 3. Autocovarianza discreta (igual que en entrenamiento)
    _, autocov_vals = autocovarianza_discreta(senal)

    # 4. FFT sobre la autocovarianza (no sobre la señal cruda)
    fft_vals = calcular_fft(autocov_vals)

    # 5. Magnitud espectral
    _, magnitud = calcular_magnitud(fft_vals, SAMPLE_RATE)

    # 4. Cargar promedios
    pat_rb = np.load(RUTA_PAT_RB)
    pat_em = np.load(RUTA_PAT_EM)

    # 5. Distancia de Manhathan (diferencia absoluta)
    dist_rb = np.sum(np.abs(magnitud - pat_rb))
    dist_em = np.sum(np.abs(magnitud - pat_em))

    print(f"--- RESULTADOS DE CLASIFICACIÓN ---")
    print(f"Distancia a 'Ruido Blanco': {dist_rb:.4f}")
    print(f"Distancia a 'Emisora':      {dist_em:.4f}")

    if dist_rb < dist_em:
        clase = "RUIDO BLANCO"
    else:
        clase = "EMISORA"

    print(f"\\n> DECISIÓN: El audio ingresado es: *** {clase} ***\\n")
    return clase


def predecir_audio(ruta_audio: str) -> str:
    """
    Lee un archivo de audio del disco, calcula su espectro de magnitud
    y lo compara con los patrones entrenados usando distancia euclidiana.
    
    Parámetros
    ----------
    ruta_audio : str
        Ruta al archivo a clasificar.
        
    Retorna
    -------
    Clase predicha: "RUIDO BLANCO" o "EMISORA"
    """
    # Cargar la señal desde disco
    senal, _ = librosa.load(ruta_audio, sr=SAMPLE_RATE, mono=True)
    
    # Delegar a la lógica matemática genérica en memoria
    return predecir_senal(senal)
