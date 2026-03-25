import os
import sys
import numpy as np

import librosa

# Agregar el directorio raíz al PATH para poder importar signal_processing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from signal_processing.analisis import autocovarianza_discreta, calcular_fft, calcular_magnitud

def cargar_audio(ruta_archivo, sr=44100):
    # Cargar audio convirtiendo a mono y a una tasa de muestreo estándar
    senal, _ = librosa.load(ruta_archivo, sr=sr, mono=True)
    return senal

def procesar_carpeta(carpeta_path, sr=44100):
    espectros_magnitud = []
    
    archivos = [f for f in os.listdir(carpeta_path) if f.lower().endswith(('.mp3', '.wav', '.mp4'))]
    if not archivos:
        print(f"Advertencia: No se encontraron archivos válidos en {carpeta_path}")
        return None
        
    print(f"Procesando {len(archivos)} archivos en {carpeta_path} ...")
    
    for archivo in archivos:
        ruta = os.path.join(carpeta_path, archivo)
        try:
            # 1. Cargar señal
            senal = cargar_audio(ruta, sr=sr)
            
            # 2. Calcular autocovarianza (SIN autocorrelación)
            lags, autocomv_val = autocovarianza_discreta(senal)
            
            # 3. FFT sobre la autocovarianza
            fft_vals = calcular_fft(autocomv_val)
            
            # 4. Magnitud
            _, magnitud = calcular_magnitud(fft_vals, sample_rate=sr)
            
            espectros_magnitud.append(magnitud)
            print(f"  + Procesado {archivo}")
        except Exception as e:
            print(f"  - Error procesando {archivo}: {e}")
            
    if not espectros_magnitud:
        return None
        
    # 5. Promediar espectros
    # Alinear tamaños en caso de ligeras variaciones de duración
    min_len = min(len(m) for m in espectros_magnitud)
    espectros_alineados = [m[:min_len] for m in espectros_magnitud]
    
    espectro_promedio = np.mean(espectros_alineados, axis=0)
    return espectro_promedio

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dir_ruido = os.path.join(base_dir, 'data', 'recordings', 'Audio-ruido-blanco')
    dir_emisora = os.path.join(base_dir, 'data', 'recordings', 'Audio-informacion')
    dir_modelos = os.path.join(base_dir, 'data', 'patterns')
    
    os.makedirs(dir_modelos, exist_ok=True)
    
    print("--- FASE DE ENTRENAMIENTO ---")
    
    # Procesar Ruido Blanco
    print("\\n[1/2] Entrenando modelo Ruido Blanco...")
    promedio_ruido = procesar_carpeta(dir_ruido)
    if promedio_ruido is not None:
        ruta_guardado = os.path.join(dir_modelos, 'ruido_promedio.npy')
        np.save(ruta_guardado, promedio_ruido)
        print(f"Modelo Ruido Blanco guardado en {ruta_guardado} (Shape: {promedio_ruido.shape})")
        
    # Procesar Emisora
    print("\\n[2/2] Entrenando modelo Emisora...")
    promedio_emisora = procesar_carpeta(dir_emisora)
    if promedio_emisora is not None:
        ruta_guardado = os.path.join(dir_modelos, 'emisora_promedio.npy')
        np.save(ruta_guardado, promedio_emisora)
        print(f"Modelo Emisora guardado en {ruta_guardado} (Shape: {promedio_emisora.shape})")
        
    print("\\n¡Entrenamiento finalizado!")

if __name__ == "__main__":
    main()
