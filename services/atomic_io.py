"""Escritura atómica de texto y JSON (A9).

Un corte a mitad de ``Path.write_text`` deja el fichero truncado, y el
consumidor (que valida el JSON al cargarlo) lo rechaza: se pierde el informe o
el guion entero. Escribir en un temporal **único por escritura** (``mkstemp``) y
sustituir con ``os.replace`` —atómico en el mismo volumen— hace que el destino
nunca exista a medias: o está el contenido viejo, o el nuevo.

Vive en ``services`` y no en ``core`` a propósito: ``services`` es la capa hoja
(no importa ``core`` ni ``ui``) y ``core`` ya importa ``services``, así que este
es el único sitio donde el helper sirve a los dos sin invertir capas.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Escribe ``text`` en ``path`` de forma atómica; devuelve la ruta.

    Ante cualquier fallo (I/O, permisos) el temporal se borra y se relanza la
    excepción original: el destino queda como estaba, nunca a medias.
    """
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(
        dir=str(destino.parent), prefix=f"{destino.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, destino)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return destino


def atomic_write_json(path: str | Path, data: Any) -> Path:
    """Escribe ``data`` como JSON UTF-8 indentado, de forma atómica."""
    return atomic_write_text(
        path, json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    )
