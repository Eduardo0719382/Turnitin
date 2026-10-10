"""Rasgos por oracion y de contexto (vecindad), y el clasificador logistico que los combina."""
import json
import re

import numpy as np

from texto import palabras, tiene_cita

_CON = re.compile(
    r"^(asimismo|además|sin embargo|por ello|por lo tanto|por tanto|en este sentido|de este modo|de esta manera|así|"
    r"por su parte|en consecuencia|no obstante|finalmente|en suma|por otro lado|por otra parte|en ese sentido|a su vez|"
    r"en cambio|de manera|al respecto|en conjunto|por consiguiente|en esta línea|en este marco|esto|esta|este|estos|"
    r"estas|su|sus|dicha|dicho)\b", re.I)
_NOM = re.compile(r"(ción|ciones|sión|siones|dad|dades|miento|mientos|ismo|encia|ancia|tivo|tiva|tivos|tivas)$", re.I)
_FUNC = {"de", "del", "la", "el", "los", "las"}
_COORD = {"y", "e", "o", "ni"}

NOMBRES = ["lp", "lr", "curv", "h", "top1", "stdlp", "tokpal", "npal", "lonpal", "largas", "nom", "mente",
           "comas", "cita", "comillas", "num", "paren", "con", "dospuntos", "mayus", "coord", "que", "func"]
VECINDADES = (1, 2, 5, 9)  # radios (en oraciones) de las ventanas de contexto


def elegible(o):
    """Oraciones que el detector puntua: texto corrido (no tablas, indice ni referencias)."""
    return o["tipo"] in ("prosa", "lista") or (o["tipo"] == "titulo" and o["npal"] >= 12)


def rasgos_oracion(o, s):
    t = o["texto"]
    ws = palabras(t)
    n = max(len(ws), 1)
    nt = max(s["n"], 1)
    low = [w.lower() for w in ws]
    return [
        s["lp"] / nt,                                   # log-probabilidad media (−log perplejidad)
        s["lr"] / nt,                                   # log-rango medio
        (s["lp"] + s["h"]) / np.sqrt(s["var"] + 1e-9),  # curvatura (Fast-DetectGPT analitico)
        s["h"] / nt,                                    # entropia media
        s["top1"] / nt,
        float(np.std(s["lps"])) if s["lps"] else 0.0,   # rafaga interna de sorpresa
        s["n"] / n,
        np.log(n),
        float(np.mean([len(w) for w in ws])) if ws else 0.0,
        sum(len(w) >= 10 for w in ws) / n,
        sum(bool(_NOM.search(w)) for w in ws) / n,
        sum(w.endswith("mente") for w in low) / n,
        t.count(",") / n,
        float(tiene_cita(t)),
        float(any(c in t for c in '"“”«»')),
        sum(w.isdigit() for w in ws) / n,
        t.count("(") / n,
        float(bool(_CON.match(t))),
        float(":" in t or ";" in t),
        sum(w[0].isupper() for w in ws[1:]) / n,
        sum(w in _COORD for w in low) / n,
        sum(w == "que" for w in low) / n,
        sum(w in _FUNC for w in low) / n,
    ]


def matriz(oraciones, stats):
    """Matriz de rasgos de las oraciones elegibles: rasgos propios + medias y dispersion en su vecindad."""
    idx = [i for i, o in enumerate(oraciones) if elegible(o)]
    X = np.array([rasgos_oracion(oraciones[i], stats[i]) for i in idx])
    w = np.array([oraciones[i]["npal"] for i in idx], float) + 1
    n = len(idx)
    cols = [X]
    j_npal, j_lp = NOMBRES.index("npal"), NOMBRES.index("lp")
    for k in VECINDADES:
        med = np.zeros_like(X)
        disp = np.zeros((n, 2))
        for j in range(n):
            a, b = max(0, j - k), min(n, j + k + 1)
            med[j] = np.average(X[a:b], axis=0, weights=w[a:b])
            disp[j] = X[a:b, j_npal].std(), X[a:b, j_lp].std()  # rafaga: variacion de longitud y de perplejidad
        cols += [med, disp]
    return idx, np.hstack(cols), w


class Modelo:
    """Regresion logistica estandarizada, guardada como JSON."""

    def __init__(self, media, escala, coef, b, umbral, suav=1, hueco=25, minimo=15):
        self.media, self.escala, self.coef, self.b = map(np.asarray, (media, escala, coef, b))
        self.umbral, self.suav, self.hueco, self.minimo = umbral, suav, hueco, minimo

    @classmethod
    def entrenar(cls, X, y, w, C=0.03):
        from sklearn.linear_model import LogisticRegression
        media, escala = X.mean(0), X.std(0) + 1e-9
        lr = LogisticRegression(C=C, max_iter=5000).fit((X - media) / escala, y, sample_weight=w)
        return cls(media, escala, lr.coef_[0], lr.intercept_[0], 0.5)

    def proba(self, X):
        z = ((X - self.media) / self.escala) @ self.coef + self.b
        return 1 / (1 + np.exp(-z))

    def guardar(self, ruta):
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        with open(ruta, "w", encoding="utf8") as f:
            json.dump(d, f)

    @classmethod
    def cargar(cls, ruta):
        with open(ruta, encoding="utf8") as f:
            return cls(**json.load(f))


def marcar(p, w, parr, umbral, suav=1, hueco=25, minimo=15):
    """Convierte probabilidades por oracion en marcas (0/1) contiguas, como los segmentos de Turnitin.

    suav: radio de la media movil; hueco: tramos sin marcar de menos de `hueco` palabras entre dos
    marcados se rellenan; minimo: se descartan segmentos marcados de menos de `minimo` palabras."""
    n = len(p)
    ps = np.array([np.average(p[max(0, j - suav):j + suav + 1], weights=w[max(0, j - suav):j + suav + 1])
                   for j in range(n)]) if suav else np.asarray(p)
    m = (ps >= umbral).astype(int)

    def tramos(v):
        out, a = [], 0
        for j in range(1, n + 1):
            if j == n or v[j] != v[a]:
                out.append((a, j, v[a]))
                a = j
        return out

    for a, b, v in tramos(m):
        if v == 0 and a > 0 and b < n and w[a:b].sum() < hueco:
            m[a:b] = 1
    for a, b, v in tramos(m):
        if v == 1 and w[a:b].sum() < minimo:
            m[a:b] = 0
    return m, ps
