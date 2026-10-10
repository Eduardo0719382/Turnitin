"""Interfaz grafica del detector de IA."""
import os
import queue
import threading
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, ttk

import actualizar


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Detector de IA  v{actualizar.version_local()}")
        self.geometry("760x460")
        self.minsize(620, 380)
        self.cola = queue.Queue()
        self.docx, self.pdf = tk.StringVar(), tk.StringVar()
        self.memoria = tk.BooleanVar(value=True)

        f = ttk.Frame(self, padding=12)
        f.pack(fill="both", expand=True)
        f.columnconfigure(1, weight=1)
        f.rowconfigure(4, weight=1)
        ttk.Label(f, text="Documento Word (.docx)").grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.docx).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(f, text="Elegir…", command=lambda: self._elegir(self.docx, "Word", "*.docx")).grid(row=0, column=2)
        ttk.Label(f, text="Reporte de Turnitin (.pdf, opcional)").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(f, textvariable=self.pdf).grid(row=1, column=1, sticky="ew", padx=6)
        ttk.Button(f, text="Elegir…", command=lambda: self._elegir(self.pdf, "PDF", "*.pdf")).grid(row=1, column=2)
        ttk.Checkbutton(f, variable=self.memoria,
                        text="Usar los veredictos de Turnitin ya aprendidos (memoria de reportes)"
                        ).grid(row=2, column=0, columnspan=3, sticky="w")

        b = ttk.Frame(f)
        b.grid(row=3, column=0, columnspan=3, sticky="ew", pady=10)
        self.botones = [
            ttk.Button(b, text="Analizar", command=self.analizar),
            ttk.Button(b, text="Aprender de este reporte", command=self.aprender),
            ttk.Button(b, text="Buscar actualización", command=self.buscar_actualizacion),
        ]
        for x in self.botones:
            x.pack(side="left", padx=(0, 8))
        self.barra = ttk.Progressbar(b, mode="indeterminate", length=140)
        self.barra.pack(side="right")

        self.log = tk.Text(f, height=12, wrap="word", state="disabled")
        self.log.grid(row=4, column=0, columnspan=3, sticky="nsew")
        self.after(100, self._bombear)
        self._hilo(self._aviso_version, silencioso=True)

    # ------------------------------------------------------------ utilidades
    def _elegir(self, var, nombre, patron):
        r = filedialog.askopenfilename(filetypes=[(nombre, patron)])
        if r:
            var.set(r)

    def decir(self, txt):
        self.cola.put(("log", str(txt)))

    def _bombear(self):
        try:
            while True:
                tipo, dato = self.cola.get_nowait()
                if tipo == "log":
                    self.log.config(state="normal")
                    self.log.insert("end", dato + "\n")
                    self.log.see("end")
                    self.log.config(state="disabled")
                elif tipo == "fin":
                    self.barra.stop()
                    for x in self.botones:
                        x.config(state="normal")
                elif tipo == "ui":
                    dato()
        except queue.Empty:
            pass
        self.after(100, self._bombear)

    def _hilo(self, fn, silencioso=False):
        if not silencioso:
            for x in self.botones:
                x.config(state="disabled")
            self.barra.start(12)

        def correr():
            try:
                fn()
            except BaseException as e:  # incluye SystemExit de la capa de consola
                if not silencioso:
                    self.decir(f"Error: {e}")
            finally:
                if not silencioso:
                    self.cola.put(("fin", None))
        threading.Thread(target=correr, daemon=True).start()

    def _pedir(self, con_pdf):
        d, p = self.docx.get().strip(), self.pdf.get().strip()
        if not os.path.isfile(d):
            messagebox.showwarning("Falta el documento", "Elige un archivo .docx.")
            return None
        if con_pdf and not os.path.isfile(p):
            messagebox.showwarning("Falta el reporte", "Elige el PDF del reporte de IA de Turnitin.")
            return None
        return d, (p if os.path.isfile(p) else None)

    # ------------------------------------------------------------ acciones
    def analizar(self):
        r = self._pedir(False)
        if not r:
            return
        d, p = r

        def trabajo():
            import detector
            res = detector.analizar(d, p, self.memoria.get(), self.decir)
            salida = os.path.splitext(d)[0] + " - deteccion IA.html"
            detector.escribir_html(res, salida)
            self.decir(detector.resumen(res))
            self.decir("Reporte: " + salida + "\n")
            webbrowser.open("file:///" + salida.replace("\\", "/"))
        self._hilo(trabajo)

    def aprender(self):
        r = self._pedir(True)
        if not r:
            return

        def trabajo():
            import detector
            detector.aprender([r], aviso=self.decir)
        self._hilo(trabajo)

    def _aviso_version(self):
        nueva, loc, rem = actualizar.hay_actualizacion()
        if nueva:
            self.decir(f"Hay una versión nueva ({rem}). Pulsa «Buscar actualización» para instalarla.")

    def buscar_actualizacion(self):
        def trabajo():
            self.decir("Consultando github.com/" + actualizar.REPO + " ...")
            nueva, loc, rem = actualizar.hay_actualizacion()
            if not nueva:
                self.decir(f"Ya tienes la última versión ({loc}).")
                return
            resp = queue.Queue()
            self.cola.put(("ui", lambda: resp.put(messagebox.askyesno(
                "Actualización disponible", f"Versión instalada: {loc}\nVersión nueva: {rem}\n\n¿Instalarla ahora?"))))
            if resp.get():
                self.decir(actualizar.actualizar()[1])
        self._hilo(trabajo)


if __name__ == "__main__":
    App().mainloop()
