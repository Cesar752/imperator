"""Convierte ``FLUJO_CODIFICAR_TROZOS.txt`` en un PDF legible (acentos y tablas).

El ``.txt`` es la fuente de verdad del flujo de codificación y vive junto a los
vídeos (``Videos\\STREAM``). Este script es el que lo maqueta en A4 y deja al lado
un PDF del mismo nombre: se ejecuta sólo cuando cambia el ``.txt``.

Usa matplotlib (ya presente) para no añadir dependencias nuevas::

    venv\\Scripts\\python.exe tools\\flujo_a_pdf.py

Las líneas ``===`` y ``---`` no se imprimen: marcan el título y los apartados de la
línea *siguiente*, que se pinta en negrita. El resto va en monoespaciada, para que
las tablas de cifras sigan alineadas.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

#: El .txt y el .pdf deben quedar en la misma carpeta, junto a la grabación.
FUENTE = Path(r"C:\Users\Cesar Esti\Videos\STREAM\FLUJO_CODIFICAR_TROZOS.txt")
DESTINO = FUENTE.with_suffix(".pdf")

#: A4 en pulgadas y márgenes.
ANCHO_PAGINA, ALTO_PAGINA = 8.27, 11.69
MARGEN_X, MARGEN_Y = 0.55, 0.55
#: Tipografía y alto de línea (pulgadas), medidos para que quepa el flujo completo.
TAMANO = 8.4
TAMANO_TITULO = TAMANO + 2.0
TAMANO_APARTADO = TAMANO + 1.0
ALTO_LINEA = 0.0142
#: Caracteres por línea antes de partir (la monoespaciada a este tamaño cabe holgada).
ANCHO_TEXTO = 98


def envolver(linea: str, ancho: int) -> list[str]:
    """Parte una línea larga por palabras, sin cortar palabras y conservando sangría.

    La sangría de la izquierda se repite en cada trozo; si no, las líneas largas
    indentadas (las viñetas) perdían el margen y quedaban descuadradas.
    """
    if len(linea) <= ancho:
        return [linea]
    sangria = linea[: len(linea) - len(linea.lstrip(" "))]
    palabras = linea.strip().split(" ")
    if not palabras:
        return [linea]
    salida: list[str] = []
    actual = ""
    for palabra in palabras:
        if not actual:
            actual = palabra
        elif len(sangria) + len(actual) + 1 + len(palabra) <= ancho:
            actual += " " + palabra
        else:
            salida.append(sangria + actual)
            actual = palabra
    if actual:
        salida.append(sangria + actual)
    return salida


def _es_separador(linea: str, caracter: str) -> bool:
    desnuda = linea.strip()
    return bool(desnuda) and set(desnuda) == {caracter}


def escribir(pdf: PdfPages, texto: str) -> None:
    """Maqueta el texto en tantas páginas como haga falta."""
    lineas = texto.splitlines()
    figura = plt.figure(figsize=(ANCHO_PAGINA, ALTO_PAGINA))
    y = ALTO_PAGINA - MARGEN_Y

    def nueva_pagina() -> None:
        nonlocal figura, y
        pdf.savefig(figura)
        plt.close(figura)
        figura = plt.figure(figsize=(ANCHO_PAGINA, ALTO_PAGINA))
        y = ALTO_PAGINA - MARGEN_Y

    for indice, original in enumerate(lineas):
        if _es_separador(original, "=") or _es_separador(original, "-"):
            continue
        siguiente = lineas[indice + 1] if indice + 1 < len(lineas) else ""
        if _es_separador(siguiente, "="):
            tamano, peso = TAMANO_TITULO, "bold"
        elif _es_separador(siguiente, "-"):
            tamano, peso = TAMANO_APARTADO, "bold"
        else:
            tamano, peso = TAMANO, "normal"
        for trozo in envolver(original.rstrip(), ANCHO_TEXTO):
            if y < MARGEN_Y:
                nueva_pagina()
            figura.text(
                MARGEN_X,
                y,
                trozo,
                fontsize=tamano,
                fontweight=peso,
                family="monospace",
                va="top",
            )
            y -= ALTO_LINEA
    pdf.savefig(figura)
    plt.close(figura)


def main() -> int:
    if not FUENTE.is_file():
        print(f"no existe el txt: {FUENTE}")
        return 1
    texto = FUENTE.read_text(encoding="utf-8")
    with PdfPages(DESTINO) as pdf:
        escribir(pdf, texto)
    print(f"pdf: {DESTINO} ({DESTINO.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
