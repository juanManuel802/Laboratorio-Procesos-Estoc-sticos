import sys

from classifier.entrenamiento import entrenar
from classifier.clasificador import predecir_audio

def mostrar_menu():
    print("=" * 50)
    print(" SISTEMA DE CLASIFICACIÓN DE SEÑALES DE AUDIO ")
    print("=" * 50)
    print("Uso:")
    print("  python main.py gui              -> Lanza la interfaz gráfica")
    print("  python main.py train            -> Inicia la fase de entrenamiento")
    print("=" * 50)

def main():
    if len(sys.argv) < 2:
        mostrar_menu()
        sys.exit(0)
        
    comando = sys.argv[1].lower()
    
    if comando == "gui":
        import pyqtgraph as pg
        from PyQt6.QtWidgets import QApplication
        from gui.main_window import VentanaPrincipal
        
        pg.setConfigOptions(antialias=True)
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        ventana = VentanaPrincipal()
        ventana.show()
        sys.exit(app.exec())
        
    elif comando == "train":
        print("Iniciando entrenamiento...")
        entrenar()

    else:
        print(f"Error: comando '{comando}' no reconocido.")
        mostrar_menu()
        sys.exit(1)

if __name__ == "__main__":
    main()
