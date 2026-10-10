"""Busqueda e instalacion de actualizaciones desde el repositorio de GitHub."""
import io
import os
import shutil
import subprocess
import urllib.request
import zipfile

REPO = "Eduardo0719382/Turnitin"
RAMA = "main"
AQUI = os.path.dirname(os.path.abspath(__file__))
NO_TOCAR = ("datos", "cache", ".git")  # lo aprendido por el usuario nunca se sobrescribe


def version_local():
    with open(os.path.join(AQUI, "VERSION"), encoding="utf8") as f:
        return f.read().strip()


def _num(v):
    return tuple(int(x) for x in v.strip().split(".") if x.isdigit())


def _git(*args):
    return subprocess.run(["git", "-C", AQUI, *args], capture_output=True, text=True, encoding="utf8",
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _es_clon():
    return os.path.isdir(os.path.join(AQUI, ".git")) and shutil.which("git")


def version_remota():
    if _es_clon():
        r = _git("fetch", "--quiet", "origin", RAMA)
        if r.returncode == 0:
            r = _git("show", f"origin/{RAMA}:VERSION")
            if r.returncode == 0:
                return r.stdout.strip()
    url = f"https://raw.githubusercontent.com/{REPO}/{RAMA}/VERSION"
    with urllib.request.urlopen(url, timeout=15) as f:
        return f.read().decode("utf8").strip()


def hay_actualizacion():
    """(hay_nueva, version_local, version_remota)"""
    loc, rem = version_local(), version_remota()
    return _num(rem) > _num(loc), loc, rem


def actualizar():
    """Instala la ultima version. Devuelve (actualizado, mensaje)."""
    try:
        nueva, loc, rem = hay_actualizacion()
    except Exception as e:
        return False, f"No se pudo consultar el repositorio: {e}"
    if not nueva:
        return False, f"Ya tienes la última versión ({loc})."
    if _es_clon():
        r = _git("pull", "--ff-only", "origin", RAMA)
        if r.returncode != 0:
            return False, "git pull falló:\n" + (r.stderr or r.stdout)
    else:
        url = f"https://github.com/{REPO}/archive/refs/heads/{RAMA}.zip"
        with urllib.request.urlopen(url, timeout=60) as f:
            z = zipfile.ZipFile(io.BytesIO(f.read()))
        for info in z.infolist():
            rel = info.filename.split("/", 1)[1] if "/" in info.filename else ""
            if not rel or info.is_dir() or rel.split("/")[0] in NO_TOCAR or ".." in rel:
                continue
            dest = os.path.join(AQUI, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as out:
                out.write(z.read(info))
    return True, f"Actualizado de {loc} a {rem}. Cierra y vuelve a abrir el programa."
