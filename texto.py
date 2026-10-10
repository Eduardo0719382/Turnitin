"""Lectura de .docx, segmentacion en oraciones y lectura de las marcas de un reporte de Turnitin."""
import re
import unicodedata
import difflib

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

CIAN = (0.32, 0.78, 0.86)  # color de relleno de las marcas "generado con IA" en el PDF de Turnitin

ABREV = r"(?:et al|al|p|pp|pág|págs|cf|cfr|ed|eds|vol|núm|n|art|cap|fig|etc|vs|Dr|Dra|Sr|Sra|Lic|Mg|aprox|trad|coord|comp|párr|s|f|min|seg|op|cit|ej|vid)"
_RE_FIN = re.compile(r"(?<=[.!?…])[\"”»’)\]]*\s+(?=[\"“«‘(\[¿¡]*[A-ZÁÉÍÓÚÑ0-9])")
_RE_ABREV = re.compile(r"(?:\b" + ABREV + r"|\b[A-ZÁÉÍÓÚÑ])\.[\"”»’)\]]*$")
_RE_CITA = re.compile(r"\((?:[^()]*?\b(?:19|20)\d{2}[a-z]?[^()]*?)\)|\b(?:19|20)\d{2}[a-z]?\)")
_RE_PAL = re.compile(r"[0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")


def norm(w):
    """Palabra en minusculas, sin tildes ni signos (para alinear docx <-> pdf)."""
    w = unicodedata.normalize("NFKD", w)
    return "".join(c for c in w if c.isalnum()).lower()


def palabras(s):
    return _RE_PAL.findall(s)


def partir_oraciones(texto):
    partes, ini = [], 0
    for m in _RE_FIN.finditer(texto):
        cand = texto[ini:m.start()]
        if _RE_ABREV.search(cand):
            continue
        # no cortar dentro de un parentesis abierto: (Autor, 2020. Otro, 2021)
        if cand.count("(") > cand.count(")"):
            continue
        partes.append(cand.strip())
        ini = m.end()
    resto = texto[ini:].strip()
    if resto:
        partes.append(resto)
    return [p for p in partes if p]


def _bloques(doc):
    """Parrafos del cuerpo en orden de lectura, incluyendo los de las tablas."""
    def rec(parent, en_tabla):
        for ch in parent.iterchildren():
            tag = ch.tag.rsplit("}", 1)[-1]
            if tag == "p":
                yield Paragraph(ch, doc), en_tabla
            elif tag == "tbl":
                t = Table(ch, doc)
                vistos = set()
                for row in t.rows:
                    for cell in row.cells:
                        if id(cell._tc) in vistos:
                            continue
                        vistos.add(id(cell._tc))
                        yield from rec(cell._tc, True)
            elif tag == "sdt":
                for c in ch.iterchildren():
                    if c.tag.endswith("sdtContent"):
                        yield from rec(c, en_tabla)
    yield from rec(doc.element.body, False)


def _texto_parrafo(p):
    """Texto completo del parrafo, incluidas las citas insertadas como controles de contenido (Mendeley, Zotero)."""
    out = []
    for el in p._p.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "t":
            out.append(el.text or "")
        elif tag in ("tab", "br", "cr"):
            out.append(" ")
    return "".join(out)


_RE_REFS = re.compile(r"^\s*(referencias|bibliograf[ií]a|referencias bibliogr[aá]ficas)\s*$", re.I)
_RE_ANEXO = re.compile(r"^\s*(ap[eé]ndices?|anexos?)\b", re.I)


def leer_docx(ruta):
    """Devuelve la lista de oraciones del documento con sus metadatos.

    Cada oracion: dict(texto, parr, tipo) donde tipo es:
      'prosa'   parrafo normal de texto corrido
      'titulo'  encabezado
      'lista'   item de lista o parrafo corto
      'tabla'   texto dentro de una tabla
      'ref'     lista de referencias
      'indice'  tabla de contenido
    """
    doc = docx.Document(ruta)
    oraciones, en_refs = [], False
    for ip, (p, en_tabla) in enumerate(_bloques(doc)):
        txt = re.sub(r"\s+", " ", _texto_parrafo(p)).strip()
        if not txt:
            continue
        try:
            estilo = (p.style.name or "").lower()
        except Exception:
            estilo = ""
        es_titulo = estilo.startswith(("heading", "título", "titulo", "title", "subt"))
        if _RE_REFS.match(txt):
            en_refs = True
        elif en_refs and (_RE_ANEXO.match(txt) and len(txt) < 60):
            en_refs = False
        es_lista = p._p.pPr is not None and p._p.pPr.numPr is not None
        if estilo.startswith(("toc", "tdc", "table of", "tabla de ilus")):
            tipo = "indice"
        elif es_titulo:
            tipo = "titulo"
        elif en_refs:
            tipo = "ref"
        elif en_tabla:
            tipo = "tabla"
        elif es_lista:
            tipo = "lista"
        else:
            tipo = "prosa"
        ors = partir_oraciones(txt) if tipo in ("prosa", "lista", "tabla") else [txt]
        for s in ors:
            oraciones.append({"texto": s, "parr": ip, "tipo": tipo, "npal": len(palabras(s))})
    return oraciones


# ---------------------------------------------------------------- reporte de Turnitin

def leer_reporte(ruta_pdf):
    """Palabras del reporte PDF de Turnitin con etiqueta 1 si estan resaltadas como IA."""
    import fitz
    d = fitz.open(ruta_pdf)
    res, pct = [], None
    for i, pg in enumerate(d):
        if i == 1:
            m = re.search(r"(\d+)\s*%\s*detectado", pg.get_text())
            if m:
                pct = int(m.group(1))
        if i < 2:
            continue
        rects = [fitz.Rect(dr["rect"]) for dr in pg.get_drawings()
                 if dr.get("fill") and all(abs(a - b) < 0.04 for a, b in zip(dr["fill"], CIAN))]
        alto = pg.rect.height
        for w in pg.get_text("words"):
            r = fitz.Rect(w[:4])
            if r.y0 < 45 or r.y1 > alto - 40:  # cabecera y pie que agrega Turnitin
                continue
            lab = 0
            for c in rects:
                inter = r & c
                if not inter.is_empty and inter.get_area() > 0.4 * r.get_area():
                    lab = 1
                    break
            n = norm(w[4])
            if n:
                res.append((n, lab))
    return res, pct


def etiquetar(oraciones, palabras_pdf):
    """Asigna a cada oracion las marcas del reporte: 'yw' (0/1 por palabra) e 'y' (fraccion marcada)."""
    a, pos = [], []
    for i, o in enumerate(oraciones):
        ws = o["texto"].split()
        o["yw"] = [None] * len(ws)
        for j, w in enumerate(ws):
            n = norm(w)
            if n:
                a.append(n)
                pos.append((i, j))
    b = [w for w, _ in palabras_pdf]
    lab = [l for _, l in palabras_pdf]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            i, j = pos[blk.a + k]
            oraciones[i]["yw"][j] = lab[blk.b + k]
    for o in oraciones:
        vistos = [v for v in o["yw"] if v is not None]
        o["cobertura"] = len(vistos) / max(len(o["yw"]), 1)
        o["y"] = sum(vistos) / len(vistos) if vistos else 0.0
        # las palabras sin alinear heredan la marca de su vecina anterior (o la mayoria de la oracion)
        previo = int(o["y"] > 0.5)
        for j, v in enumerate(o["yw"]):
            if v is None:
                o["yw"][j] = previo
            else:
                previo = v
    return oraciones


def clave(texto):
    """Huella de una oracion (no permite reconstruir el texto)."""
    import hashlib
    return hashlib.sha1(" ".join(filter(None, map(norm, texto.split()))).encode("utf8")).hexdigest()[:16]


def tiene_cita(s):
    return bool(_RE_CITA.search(s))
