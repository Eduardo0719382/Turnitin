"""Detector de escritura con IA calibrado contra reportes de Turnitin.

Uso por consola (la interfaz grafica esta en app.py):
  python detector.py analizar documento.docx [--reporte turnitin.pdf] [-o salida.html] [--sin-memoria]
  python detector.py aprender doc1.docx reporte1.pdf [doc2.docx reporte2.pdf ...]
  python detector.py actualizar
"""
import argparse
import html
import json
import os
import sys

import numpy as np

import lm
from rasgos import Modelo, marcar, matriz
from texto import clave, etiquetar, leer_docx, leer_reporte

AQUI = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join(AQUI, "datos")  # lo que aprende cada usuario; no se sube al repositorio
CALIFICABLE = ("prosa", "lista", "titulo", "ref")
MIN_PAL = 5  # las oraciones mas cortas no cuentan para el porcentaje


def _ruta(nombre, local):
    return os.path.join(DATOS if local else AQUI, nombre)


def _leer_json(nombre, local):
    r = _ruta(nombre, local)
    if os.path.exists(r):
        with open(r, encoding="utf8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------- memoria de reportes

def cargar_memoria():
    """Veredictos de Turnitin ya vistos: huella de oracion -> '0', '1' o cadena de bits por palabra."""
    mem = {}
    for local in (False, True):
        mem.update(_leer_json("memoria.json", local) or {})
    return mem


def _memorizar(mem, ors):
    previa = ""
    for o in ors:
        k = clave(o["texto"])
        bits = "".join(map(str, o["yw"]))
        v = bits if 0 < sum(o["yw"]) < len(o["yw"]) else bits[:1]
        if v:
            mem[previa + ">" + k] = v  # con contexto: distingue oraciones repetidas
            mem.setdefault(k, v)
        previa = k


def _recordar(mem, ors):
    """Para cada oracion: lista de marcas por palabra si Turnitin ya la evaluo, o None."""
    out, previa = [], ""
    for o in ors:
        k = clave(o["texto"])
        n = len(o["texto"].split())
        v = mem.get(previa + ">" + k) or mem.get(k)
        if v is None or (len(v) > 1 and len(v) != n):
            out.append(None)
        else:
            out.append([int(c) for c in v] if len(v) > 1 else [int(v)] * n)
        previa = k
    return out


# ---------------------------------------------------------------- modelo estadistico

def preparar(ruta_docx, ruta_pdf=None, aviso=print):
    ors = leer_docx(ruta_docx)
    pct = None
    if ruta_pdf:
        pw, pct = leer_reporte(ruta_pdf)
        etiquetar(ors, pw)
    aviso(f"Calculando perplejidad de {os.path.basename(ruta_docx)} (1-2 min la primera vez)...")
    st = lm.stats_oraciones([o["texto"] for o in ors])
    idx, X, w = matriz(ors, st)
    d = dict(ors=ors, idx=idx, X=X, w=w, pct=pct, ruta=ruta_docx)
    if ruta_pdf:
        d["y"] = np.array([ors[i]["y"] > 0.5 for i in idx]).astype(int)
    return d


def _f1(y, m, w):
    tp = (w * (y & m)).sum()
    return 2 * tp / ((w * y).sum() + (w * m).sum() + 1e-9)


def _ajustar_post(modelo, docs):
    """Elige umbral / suavizado / relleno que maximizan el F1 por palabra sobre los documentos dados."""
    mejor = (-1, None)
    ps = [modelo.proba(d["X"]) for d in docs]
    for suav in (0, 1, 2):
        for hueco in (0, 15, 30, 60):
            for minimo in (0, 12, 25):
                for u in np.arange(0.15, 0.86, 0.025):
                    f = np.mean([_f1(d["y"], marcar(p, d["w"], None, u, suav, hueco, minimo)[0], d["w"])
                                 for d, p in zip(docs, ps)])
                    if f > mejor[0]:
                        mejor = (f, (float(u), suav, hueco, minimo))
    modelo.umbral, modelo.suav, modelo.hueco, modelo.minimo = mejor[1]
    return modelo


def _fit(docs, C=3.0):
    X = np.vstack([d["X"] for d in docs])
    y = np.concatenate([d["y"] for d in docs])
    w = np.concatenate([d["w"] for d in docs])
    return _ajustar_post(Modelo.entrenar(X, y, w, C=C), docs)


def cargar_modelo():
    for local in (True, False):
        if os.path.exists(_ruta("modelo.json", local)):
            return Modelo.cargar(_ruta("modelo.json", local))
    raise SystemExit("Falta modelo.json")


def _cargar_entrenamiento():
    """Matrices de rasgos (sin texto) de todos los documentos aprendidos: los del repositorio y los locales."""
    docs = {}
    for local in (False, True):
        r = _ruta("entrenamiento.npz", local)
        if os.path.exists(r):
            z = np.load(r)
            for k in z.files:
                if k.startswith("X_"):
                    docs[k[2:]] = dict(X=z[k], y=z["y_" + k[2:]], w=z["w_" + k[2:]])
    return docs


# ---------------------------------------------------------------- analisis

def _tramos(v):
    return int((np.diff(np.concatenate([[0], np.asarray(v, int)])) == 1).sum())


def analizar(ruta_docx, ruta_pdf=None, usar_memoria=True, aviso=print):
    d = preparar(ruta_docx, ruta_pdf, aviso)
    ors = d["ors"]
    marc, ps = marcar(cargar_modelo().proba(d["X"]), d["w"], None, *_post(cargar_modelo()))
    mod = {i: (int(v), float(p)) for i, v, p in zip(d["idx"], marc, ps)}
    rec = _recordar(cargar_memoria(), ors) if usar_memoria else [None] * len(ors)
    conocidas = 0
    for i, o in enumerate(ors):
        n = len(o["texto"].split())
        o["p"] = mod.get(i, (0, 0.0))[1]
        if rec[i] is not None:
            o["m"], o["origen"] = rec[i], "turnitin"
            conocidas += n
        else:
            o["m"], o["origen"] = [mod.get(i, (0, 0.0))[0]] * n, "modelo"
    m = np.array([b for o in ors for b in o["m"]])
    cal = sum(len(o["m"]) for o in ors if o["tipo"] in CALIFICABLE and len(o["m"]) >= MIN_PAL)
    res = dict(d=d, ors=ors, pct=100 * m.sum() / max(cal, 1), segmentos=_tramos(m),
               conocido=conocidas / max(len(m), 1), pct_turnitin=d["pct"])
    if ruta_pdf:
        y = np.array([b for o in ors for b in o["yw"]])
        tp, fp, fn = ((y == 1) & (m == 1)).sum(), ((y == 0) & (m == 1)).sum(), ((y == 1) & (m == 0)).sum()
        res.update(acuerdo=100 * (y == m).mean(), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                   f1=2 * tp / max(2 * tp + fp + fn, 1), segmentos_turnitin=_tramos(y))
    return res


def _post(m):
    return m.umbral, m.suav, m.hueco, m.minimo


CSS = """body{font:16px/1.6 Georgia,serif;max-width:860px;margin:2em auto;padding:0 16px;color:#222}
h1{font:600 22px system-ui}.res{font:15px system-ui;background:#f3f6f8;padding:12px 16px;border-radius:8px}
.big{font-size:34px;font-weight:700}.ia{background:#52c7db}.tt{background:#ffd27a}.am{background:#52c7db}
.err .am{border-bottom:3px solid #2a8a3a}.t{font-weight:700;margin-top:1.2em}.x{color:#888}p{margin:.6em 0}
.ley span{padding:1px 6px;margin-right:10px}.nota{color:#555;font-size:13px;margin-top:6px}"""


def escribir_html(res, salida):
    comp = "acuerdo" in res
    out = [f"<!doctype html><meta charset=utf-8><title>Detector IA</title><style>{CSS}</style>",
           f"<h1>{html.escape(os.path.basename(res['d']['ruta']))}</h1><div class=res>",
           f"<div class=big>{res['pct']:.0f} % detectado como IA</div>{res['segmentos']} segmentos marcados"]
    if comp:
        out.append(f"<br>Turnitin: {res['pct_turnitin']} % ({res['segmentos_turnitin']} segmentos) · coincidencia "
                   f"por palabra {res['acuerdo']:.1f} % · precisión {res['precision']:.2f} · recall "
                   f"{res['recall']:.2f}<div class=ley><span class=am>coincide con Turnitin</span>"
                   "<span class=ia style='outline:2px solid #d9534f'>solo este detector</span>"
                   "<span class=tt>solo Turnitin</span></div>")
    else:
        out.append("<div class=ley><span class=ia>probable IA</span></div>")
    out.append(f"<div class=nota>{100 * res['conocido']:.0f} % del texto ya fue evaluado por Turnitin en un reporte "
               "aprendido (se usa su veredicto exacto); el resto es una estimación del modelo.</div></div>")
    parr = None
    for o in res["ors"]:
        if o["parr"] != parr:
            if parr is not None:
                out.append("</p>")
            out.append("<p class=%s>" % {"titulo": "t", "ref": "x", "indice": "x", "tabla": "x"}.get(o["tipo"], "n"))
            parr = o["parr"]
        ws = o["texto"].split()
        yw = o.get("yw") if comp else None
        j = 0
        while j < len(ws):
            est = (o["m"][j], yw[j] if yw else 0)
            k = j
            while k < len(ws) and (o["m"][k], yw[k] if yw else 0) == est:
                k += 1
            txt = html.escape(" ".join(ws[j:k]))
            if est == (1, 1):
                txt = f"<span class=am>{txt}</span>"
            elif est[0]:
                extra = " style='outline:2px solid #d9534f'" if comp else ""
                txt = f"<span class=ia{extra} title='p={o['p']:.2f} ({o['origen']})'>{txt}</span>"
            elif est[1]:
                txt = f"<span class=tt>{txt}</span>"
            out.append(txt)
            j = k
    out.append("</p>")
    with open(salida, "w", encoding="utf8") as f:
        f.write("\n".join(out))
    return salida


def resumen(res):
    s = f"{res['pct']:.0f} % detectado como IA, {res['segmentos']} segmentos"
    if "acuerdo" in res:
        s += (f"\nTurnitin: {res['pct_turnitin']} %, {res['segmentos_turnitin']} segmentos | coincidencia por palabra "
              f"{res['acuerdo']:.1f} %  precisión {res['precision']:.2f}  recall {res['recall']:.2f}")
    return s + f"\n{100 * res['conocido']:.0f} % del texto ya evaluado por Turnitin en reportes aprendidos"


# ---------------------------------------------------------------- aprendizaje

def aprender(pares, publicar=False, aviso=print):
    """Incorpora pares (docx, pdf de Turnitin): memoriza sus marcas y reentrena el modelo con todo lo aprendido."""
    local = not publicar
    os.makedirs(DATOS, exist_ok=True)
    mem = _leer_json("memoria.json", local) or {}
    nuevos = {}
    for dx, pdf in pares:
        d = preparar(dx, pdf, aviso)
        if d["pct"] is None:
            raise ValueError(f"{os.path.basename(pdf)} no parece un reporte de IA de Turnitin")
        _memorizar(mem, d["ors"])
        nuevos[clave(os.path.basename(dx))] = d
    with open(_ruta("memoria.json", local), "w", encoding="utf8") as f:
        json.dump(mem, f)
    docs = _cargar_entrenamiento()
    propios = {}
    r = _ruta("entrenamiento.npz", local)
    if os.path.exists(r):
        z = np.load(r)
        propios = {k: z[k] for k in z.files}
    for k, d in nuevos.items():
        docs[k] = d
        propios.update({"X_" + k: d["X"], "y_" + k: d["y"], "w_" + k: d["w"]})
    np.savez_compressed(r, **propios)
    aviso(f"Reentrenando con {len(docs)} documentos...")
    lista = list(docs.values())
    if len(lista) > 1:
        aviso("Validación cruzada (modelo sin memoria, entrenado sin el documento evaluado):")
        for k, d in nuevos.items():
            m = _fit([x for kk, x in docs.items() if kk != k])
            marc, _ = marcar(m.proba(d["X"]), d["w"], None, *_post(m))
            aviso(f"  {os.path.basename(d['ruta'])[:50]}: F1 {_f1(d['y'], marc, d['w']):.2f}")
    _fit(lista).guardar(_ruta("modelo.json", local))
    aviso("Modelo y memoria actualizados.")


# ---------------------------------------------------------------- consola

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("analizar")
    s.add_argument("documento")
    s.add_argument("--reporte", help="PDF del reporte de IA de Turnitin, para comparar")
    s.add_argument("-o", "--salida")
    s.add_argument("--sin-memoria", action="store_true", help="usar solo el modelo, sin veredictos memorizados")
    s = sub.add_parser("aprender")
    s.add_argument("archivos", nargs="+")
    s.add_argument("--publicar", action="store_true", help="guardar en los archivos del repositorio, no en datos/")
    sub.add_parser("actualizar")
    a = ap.parse_args()
    err = lambda t: print(t, file=sys.stderr)
    if a.cmd == "analizar":
        res = analizar(a.documento, a.reporte, not a.sin_memoria, err)
        salida = a.salida or os.path.splitext(a.documento)[0] + " - deteccion IA.html"
        escribir_html(res, salida)
        print(resumen(res))
        print("Reporte:", salida)
    elif a.cmd == "aprender":
        if len(a.archivos) % 2:
            sys.exit("Se esperan pares: documento.docx reporte.pdf")
        aprender(list(zip(a.archivos[::2], a.archivos[1::2])), a.publicar)
    else:
        import actualizar
        print(actualizar.actualizar()[1])


if __name__ == "__main__":
    main()
