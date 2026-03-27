import os
import numpy as np
from signal_processing.analisis import normalizar, calcular_espectro_directo

SAMPLE_RATE  = 44100
DURACION_SEG = 2
N_MUESTRAS   = SAMPLE_RATE * DURACION_SEG
FREQ_MAX     = 8000

CARPETA_PAT   = os.path.join(os.path.dirname(__file__), "..", "data", "patterns")
RUTA_STATS_RB = os.path.join(CARPETA_PAT, "stats_rb.npy")
RUTA_STATS_EM = os.path.join(CARPETA_PAT, "stats_em.npy")

def _recortar(espectro: np.ndarray) -> np.ndarray:
    idx_max = int(FREQ_MAX * len(espectro) / (SAMPLE_RATE / 2))
    return espectro[:idx_max]

def _planitud_espectral(espectro: np.ndarray) -> float:
    espectro = espectro + 1e-10
    return float(np.exp(np.mean(np.log(espectro))) / np.mean(espectro))

def _varianza_espectral(espectro: np.ndarray) -> float:
    espectro = espectro + 1e-10
    return float(np.var(np.log10(espectro)))

def predecir_senal(senal: np.ndarray) -> str:
    if not os.path.exists(RUTA_STATS_RB) or not os.path.exists(RUTA_STATS_EM):
        raise FileNotFoundError("Sin patrones. Corre: python main.py train")

    stats_rb = np.load(RUTA_STATS_RB, allow_pickle=True).item()
    stats_em = np.load(RUTA_STATS_EM, allow_pickle=True).item()

    if len(senal) > N_MUESTRAS:
        senal = senal[:N_MUESTRAS]
    elif len(senal) < N_MUESTRAS:
        senal = np.pad(senal, (0, N_MUESTRAS - len(senal)))

    senal    = normalizar(senal)
    espectro = _recortar(calcular_espectro_directo(senal))

    planitud = _planitud_espectral(espectro)
    varianza = _varianza_espectral(espectro)

    def distancia(valor, media, std):
        std_ef = max(std, 0.05)
        return abs(valor - media) / std_ef

    dist_rb = (distancia(planitud, stats_rb["planitud_media"], stats_rb["planitud_std"]) +
               distancia(varianza, stats_rb["varianza_media"], stats_rb["varianza_std"]))
    dist_em = (distancia(planitud, stats_em["planitud_media"], stats_em["planitud_std"]) +
               distancia(varianza, stats_em["varianza_media"], stats_em["varianza_std"]))

    print("--- RESULTADOS DE CLASIFICACIÓN ---")
    print(f"Planitud: {planitud:.6f}  (RB: {stats_rb['planitud_media']:.4f} | EM: {stats_em['planitud_media']:.4f})")
    print(f"Varianza: {varianza:.6f}  (RB: {stats_rb['varianza_media']:.4f} | EM: {stats_em['varianza_media']:.4f})")
    print(f"Dist RB:  {dist_rb:.4f} | Dist EM: {dist_em:.4f}")

    clase = "RUIDO BLANCO" if dist_rb < dist_em else "EMISORA"
    print(f"\n> DECISIÓN: *** {clase} ***\n")
    return clase