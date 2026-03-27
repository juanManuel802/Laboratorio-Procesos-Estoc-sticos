import os
import numpy as np
from signal_processing.analisis import normalizar_rms, autocovarianza_discreta, calcular_fft, calcular_magnitud

SAMPLE_RATE    = 44100
DURACION_SEG   = 2
N_MUESTRAS     = SAMPLE_RATE * DURACION_SEG

CARPETA_PAT    = os.path.join(os.path.dirname(__file__), "..", "data", "patterns")
RUTA_PAT_RB    = os.path.join(CARPETA_PAT, "espectro_promedio_rb.npy")
RUTA_PAT_EM    = os.path.join(CARPETA_PAT, "espectro_promedio_em.npy")

def predecir_senal(senal: np.ndarray) -> str:
    if not os.path.exists(RUTA_PAT_RB) or not os.path.exists(RUTA_PAT_EM):
        raise FileNotFoundError("No se encontraron los patrones. Corre el entrenamiento primero.")
    
    if len(senal) > N_MUESTRAS:
        senal = senal[:N_MUESTRAS]
    elif len(senal) < N_MUESTRAS:
        senal = np.pad(senal, (0, N_MUESTRAS - len(senal)))

    senal = normalizar_rms(senal)
    autocov_vals = autocovarianza_discreta(senal)
    fft_vals = calcular_fft(autocov_vals)
    magnitud = calcular_magnitud(fft_vals, SAMPLE_RATE)

    pat_rb = np.load(RUTA_PAT_RB)
    pat_em = np.load(RUTA_PAT_EM)

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
