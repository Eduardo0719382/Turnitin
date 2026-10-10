"""Estadisticas de predictibilidad por oracion con un GPT-2 en espanol (corre en CPU)."""
import hashlib
import json
import os

import numpy as np

MODELO = "flax-community/gpt-2-spanish"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
_m = {}


def _cargar():
    if not _m:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _m["tok"] = AutoTokenizer.from_pretrained(MODELO)
        _m["mod"] = AutoModelForCausalLM.from_pretrained(MODELO, use_safetensors=True).eval()
        torch.set_grad_enabled(False)
    return _m["tok"], _m["mod"]


def stats_oraciones(textos, ventana=1024, paso=512, progreso=None):
    """Para cada oracion devuelve un dict con sumas por token:
    n, lp (log-prob), lr (log-rango), h (entropia), var (varianza de log-prob), top1, top10, lps (lista).
    El contexto de cada token es el texto previo del documento (hasta `ventana` tokens)."""
    clave = hashlib.sha1(("\n".join(textos) + MODELO).encode("utf8")).hexdigest()[:20]
    ruta = os.path.join(CACHE, clave + ".json")
    if os.path.exists(ruta):
        with open(ruta, encoding="utf8") as f:
            return json.load(f)
    import torch
    tok, mod = _cargar()
    ids, dueno = [tok.bos_token_id or tok.eos_token_id], [-1]
    for i, t in enumerate(textos):
        e = tok.encode(" " + t, add_special_tokens=False)
        ids += e
        dueno += [i] * len(e)
    ids_t = torch.tensor(ids)
    N = len(ids)
    lp = np.zeros(N); lr = np.zeros(N); h = np.zeros(N); var = np.zeros(N); rk = np.zeros(N)
    ini, hecho = 0, 1  # tokens [0, hecho) ya puntuados
    while hecho < N:
        fin = min(ini + ventana, N)
        logits = mod(ids_t[ini:fin].unsqueeze(0)).logits[0]
        logp = torch.log_softmax(logits.double(), -1)
        # la posicion j predice el token ini+j+1
        a = hecho - ini - 1
        sub = logp[a:fin - ini - 1]
        obj = ids_t[hecho:fin]
        l = sub.gather(1, obj.unsqueeze(1)).squeeze(1)
        p = sub.exp()
        ent = -(p * sub).sum(1)
        v = (p * sub * sub).sum(1) - ent * ent
        r = (sub > l.unsqueeze(1)).sum(1) + 1
        lp[hecho:fin] = l.numpy(); h[hecho:fin] = ent.numpy(); var[hecho:fin] = v.numpy()
        rk[hecho:fin] = r.numpy(); lr[hecho:fin] = np.log(r.numpy())
        hecho = fin
        ini = fin - (ventana - paso)
        if progreso:
            progreso(hecho / N)
    res = [dict(n=0, lp=0.0, lr=0.0, h=0.0, var=0.0, top1=0, top10=0, lps=[]) for _ in textos]
    for k in range(1, N):
        s = res[dueno[k]]
        s["n"] += 1; s["lp"] += lp[k]; s["lr"] += lr[k]; s["h"] += h[k]; s["var"] += var[k]
        s["top1"] += int(rk[k] == 1); s["top10"] += int(rk[k] <= 10)
        s["lps"].append(round(float(lp[k]), 3))
    os.makedirs(CACHE, exist_ok=True)
    with open(ruta, "w", encoding="utf8") as f:
        json.dump(res, f)
    return res
