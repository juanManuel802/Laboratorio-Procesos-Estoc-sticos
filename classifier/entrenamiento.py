import os
import numpy as np
import soundfile as sf
from signal_processing.analisis import normalizar, calcular_espectro_directo

SAMPLE_RATE  = 44100
DURACION_SEG = 2
N_MUESTRAS   = SAMPLE_RATE * DURACION_SEG
FREQ_MAX     = 8000

CARPETA_RB  = "data/recordings/ruido_blanco"
CARPETA_EM  = "data/recordings/senal_radio"
CARPETA_PAT = "data/patterns"

def _leer_audio(ruta: str) -> np.ndarray:
    senal, sr = sf.read(ruta, dtype='float32')
    if senal.ndim == 2:
        senal = senal[:, 0]
    # Resamplear si es necesario
    if sr != SAMPLE_RATE:
        import librosa
        senal = librosa.resample(senal, orig_sr=sr, target_sr=SAMPLE_RATE)
    if len(senal) > N_MUESTRAS:
        senal = senal[:N_MUESTRAS]
    elif len(senal) < N_MUESTRAS:
        senal = np.pad(senal, (0, N_MUESTRAS - len(senal)))
    return senal

def _recortar(espectro: np.ndarray) -> np.ndarray:
    idx_max = int(FREQ_MAX * len(espectro) / (SAMPLE_RATE / 2))
    return espectro[:idx_max]

def _planitud_espectral(espectro: np.ndarray) -> float:
    espectro = espectro + 1e-10
    return float(np.exp(np.mean(np.log(espectro))) / np.mean(espectro))

def _varianza_espectral(espectro: np.ndarray) -> float:
    espectro = espectro + 1e-10
    return float(np.var(np.log10(espectro)))

def _procesar_carpeta(carpeta: str) -> dict:
    archivos = sorted([
        os.path.join(carpeta, f)
        for f in os.listdir(carpeta)
        if f.endswith(".wav")
    ])
    n = len(archivos)
    print(f"  → {n} archivos en: {carpeta}")

    n_bins = int(FREQ_MAX * (N_MUESTRAS // 2) / (SAMPLE_RATE / 2))
    matriz  = np.zeros((n, n_bins))
    planitudes = np.zeros(n)
    varianzas  = np.zeros(n)

    for i, ruta in enumerate(archivos):
        print(f"    [{i+1}/{n}] {os.path.basename(ruta)}")
        senal    = normalizar(_leer_audio(ruta))
        espectro = _recortar(calcular_espectro_directo(senal))
        matriz[i]      = espectro
        planitudes[i]  = _planitud_espectral(espectro)
        varianzas[i]   = _varianza_espectral(espectro)

    return {
        "espectro_promedio": np.mean(matriz, axis=0),
        "planitud_media": float(np.mean(planitudes)),
        "planitud_std":   float(np.std(planitudes)),
        "varianza_media": float(np.mean(varianzas)),
        "varianza_std":   float(np.std(varianzas)),
    }

def entrenar():
    os.makedirs(CARPETA_PAT, exist_ok=True)

    print("═" * 50)
    print("ENTRENAMIENTO — Ruido blanco")
    print("═" * 50)
    stats_rb = _procesar_carpeta(CARPETA_RB)

    print()
    print("═" * 50)
    print("ENTRENAMIENTO — Emisora")
    print("═" * 50)
    stats_em = _procesar_carpeta(CARPETA_EM)

    np.save(os.path.join(CARPETA_PAT, "espectro_promedio_rb.npy"), stats_rb["espectro_promedio"])
    np.save(os.path.join(CARPETA_PAT, "espectro_promedio_em.npy"), stats_em["espectro_promedio"])
    np.save(os.path.join(CARPETA_PAT, "stats_rb.npy"), stats_rb, allow_pickle=True)
    np.save(os.path.join(CARPETA_PAT, "stats_em.npy"), stats_em, allow_pickle=True)

    print()
    print("Entrenamiento completado:")
    print(f"  Ruido Blanco — planitud: {stats_rb['planitud_media']:.6f} ± {stats_rb['planitud_std']:.6f}")
    print(f"  Emisora      — planitud: {stats_em['planitud_media']:.6f} ± {stats_em['planitud_std']:.6f}")
    print(f"  Ruido Blanco — varianza: {stats_rb['varianza_media']:.6f} ± {stats_rb['varianza_std']:.6f}")
    print(f"  Emisora      — varianza: {stats_em['varianza_media']:.6f} ± {stats_em['varianza_std']:.6f}")

if __name__ == "__main__":
    entrenar()