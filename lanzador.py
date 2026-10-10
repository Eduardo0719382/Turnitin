"""Lanzador de «Detector IA.exe»: abre app.py con el Python instalado.

El .exe solo arranca el programa; el codigo sigue en los .py para que «Buscar actualización» pueda renovarlo.
Se compila con:  python -m PyInstaller --onefile --noconsole --name "Detector IA" lanzador.py
"""
import ctypes
import os
import shutil
import subprocess
import sys


def main():
    aqui = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    app = os.path.join(aqui, "app.py")
    py = shutil.which("pythonw") or shutil.which("python")
    if not py or not os.path.exists(app):
        falta = "Python 3.10 o superior (python.org)" if not py else "app.py junto a este ejecutable"
        ctypes.windll.user32.MessageBoxW(0, f"No se encontró {falta}.", "Detector de IA", 0x10)
        return
    r = subprocess.run([py, "-c", "import docx, fitz, numpy, sklearn, torch, transformers"], cwd=aqui,
                       creationflags=0x08000000)
    if r.returncode != 0:
        ctypes.windll.user32.MessageBoxW(
            0, "Se instalarán las dependencias del programa. Puede tardar varios minutos.", "Detector de IA", 0x40)
        subprocess.run([shutil.which("python") or py, "-m", "pip", "install", "-r", "requirements.txt"], cwd=aqui)
    subprocess.Popen([py, app], cwd=aqui)


if __name__ == "__main__":
    main()
