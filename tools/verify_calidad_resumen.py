"""Verificación de la **calidad** del resumen (documento real o sintético).

Mide si el resumen se corresponde con el documento, no solo cuánto tarda. Con
documentos sintéticos (apartados y cifras comprobables) reproduce los fallos ya
vistos: recorte del texto, cifras inventadas por extrapolación, cambio de
idioma. Con un PDF real comprueba lo mismo sobre el documento del usuario.

Uso::

    venv\\Scripts\\python.exe tools\\verify_calidad_resumen.py
    venv\\Scripts\\python.exe tools\\verify_calidad_resumen.py --eu
    venv\\Scripts\\python.exe tools\\verify_calidad_resumen.py --pdf "C:\\ruta\\LANALDIA.pdf"
    venv\\Scripts\\python\\python.exe tools\\verify_calidad_resumen.py --pdf doc.pdf --modelo qwen3-vl:8b

Con ``--pdf`` **no** se inventa nada: se usa el documento tal cual, su texto
embebido (o el OCR si la página no tiene capa de texto) y el motor configurado.
Muestra el resumen completo para poder leerlo, que es lo que un número no dice.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402

from core.engine_manager import EngineManager  # noqa: E402
from services import document_params as dp  # noqa: E402
from services import document_pipeline as dpl  # noqa: E402
from ui import document_widget as dw  # noqa: E402

MODELO = "qwen3-vl:8b"
PAGINAS = 19

#: Página en castellano con apartados y cifras propias, todas comprobables.
PLANTILLA_ES = """ACTA {n} DE LA COMISION DE SEGUIMIENTO

1. Presupuesto del ejercicio. La partida aprobada para {n} asciende a {importe} euros, con un remanente de {remanente} euros que se arrastra al ejercicio siguiente y que no podra destinarse a gasto corriente.

2. Personal adscrito. La plantilla queda fijada en {personas} personas, de las cuales {parcial} trabajan a tiempo parcial. Las contrataciones nuevas se limitan a {nuevas} plazas y requieren informe previo de intervencion.

3. Plazos y presentacion. Las solicitudes se presentaran antes del {dia} de octubre, con un plazo de subsanacion de {subsanacion} dias naturales. No se admitiran escritos fuera de plazo.

4. Seguimiento. La comision se reunira cada {periodicidad} meses y levantara acta con el acuerdo {acuerdo}. El acta se publicara en el tablon de anuncios.
"""

#: La misma prueba en euskera: aquí se veía el fallo del idioma.
PLANTILLA_EU = """LAN-BALDINTZEN HITZARMENA {n}

1. Lanaldia. Asteko lanaldia 40 ordukoa da batez beste, eta urteko lanaldia 1776 ordukoa. Hitzarmenaren bidez 24 lanegun kenduta murriztu daiteke.

2. Eguneko banaketa. Eguneko gehienezko lanaldia 9 ordukoa da, eta atsedenaldia gutxienez 12 ordukoa da bi lanegun artean.

3. Ordutegia. Gaueko lana 22:00etatik 06:00etara bitartekoa da, gutxienez 3 ordu, eta debekatuta dago 18 urte baino gutxiagorentzat.

4. Oporrak eta baimenak. Urtean 30 egun opor ordainduta eta 14 jaiegun daude. Ezkontzak 15 eguneko baimena ematen du, gaixotasunak 5 egun.
"""

#: Convenio laboral de una página, en euskera, con los cuatro patrones que
#: fallaban en el documento real: cifras parecidas entre apartados (24 horas /
#: 24 ordu, 12 y 7 horas de descanso), listas de permisos, y una prohibición con
#: excepción (los nocturnos no hacen horas extra salvo fuerza mayor). Reproduce
#: el caso sin necesidad del PDF del usuario.
PLANTILLA_CONVENIO = """LAN-BALDINTZEN HITZARMENA

1. Lanaldi arrunta. Jornada ordinaria es el tiempo realmente trabajado entre entrar y salir. Asteko lanaldia 40 ordukoa da, eta urteko lanaldia 1800 ordukoa. Hitzarmenaren bidez 1776 ordura arte murriztu daiteke, hau da, 24 ordu gutxiago.

2. Lanegunaren banaketa erregularra. Eguneko gehienezko lanaldia 9 ordukoa da. Atsedena lanegun artean: 12 ordu. Asteko atsedena: 1,5 egun jarraian. Barneko atsedenaldia: 15 minutu, 6 ordu baino gehiago egiten badira jarraian.

3. Lanegunaren banaketa irregularra. Enpresak urteko orduen %10 erabili dezake lanaldia luzatzeko, 5 egun lehenago jakinarazita, eta 12 hilabetetan konpentsatu behar da.

4. Lan-ordutegia. Ordutegia laneko ordu zehatzak dira, enpresaren eta langileen artean itundutakoak, eta ez dira lanaldiarekin nahastu behar.

5. Txandakako lana. Txandak goizekoa, arratsaldekoa eta gauekoa dira. 24 orduko prozesuetan, gaueko txandak ezin du bi aste baino gehiago iraun jarraian, langileak berak nahi ez badu. Atsedena txanden artean: 7 ordu.

6. Gaueko lana. Gaueko lana 22:00etatik 06:00etara bitartekoa da. Gauekotzat jotzen da tarte horretan 3 ordu egiten badira. Plus espezifikoa du. Debekatuta dago 18 urte baino gutxiagorentzat eta haurdun daudenentzat. Ezin dute aparteko ordurik egin.

7. Aparteko orduak. Aparteko orduak jornada arruntetik gorako orduak dira. Gehienez 80 ordu urtean. Debekatuta daude gaueko langileentzat, indar majarrak izan ezik.

8. Ordaindutako baimenak. Ezkontza: 15 egun. Gaixotasun larria: 5 egun. Heriotza: 2 egun. Familia-premiazkoak: 4 egun urtean. Bizitokia aldatzea: 1 egun. Hauteskunde-mahaia: egun osoa eta 5 orduko atsedena.

9. Oporrak eta jaiegunak. Oporrak: 30 egun natural urtean. Jaiegunak: 14 egun urtean. Jaiegun batean lan eginez gero: %75 gehiago edo atseden konpentsagarria.
"""

#: Palabras que delatan una respuesta en inglés.
_PALABRAS_INGLESAS = (" the ", " and ", " with ", " of ", " is ", " are ", " to ")


def idioma_sospechoso(texto: str) -> bool:
    """¿El resultado parece escrito en inglés?"""
    minusculas = f" {texto.lower()} "
    return sum(minusculas.count(palabra) for palabra in _PALABRAS_INGLESAS) >= 5


class _Contador:
    """Motor real envuelto para contar llamadas."""

    def __init__(self, real) -> None:
        self.real = real
        self.texto = 0
        self.imagen = 0

    def analyze_text(self, *args, **kwargs):
        self.texto += 1
        return self.real.analyze_text(*args, **kwargs)

    def analyze_image(self, *args, **kwargs):
        self.imagen += 1
        return self.real.analyze_image(*args, **kwargs)


def pagina(numero: int, plantilla: str) -> str:
    """Texto de la página ``numero`` con cifras únicas para poder comprobarlas."""
    return plantilla.format(
        n=numero,
        importe=100_000 + numero * 7_137,
        remanente=3_000 + numero * 517,
        personas=12 + numero,
        parcial=2 + numero % 4,
        nuevas=1 + numero % 3,
        dia=(numero % 28) + 1,
        subsanacion=10 + numero,
        periodicidad=2 + numero % 3,
        acuerdo=f"AC-{numero:03d}",
    )


def crear_pdf_sintetico(destino: Path, paginas: int, plantilla: str) -> None:
    """PDF con ``paginas`` páginas de texto embebido."""
    doc = pymupdf.open()
    for numero in range(1, paginas + 1):
        page = doc.new_page()
        page.insert_textbox(
            pymupdf.Rect(50, 50, 545, 790), pagina(numero, plantilla), fontsize=10
        )
    doc.save(str(destino))
    doc.close()


def texto_de_pdf(ruta: Path) -> list[str]:
    """Texto embebido de cada página del PDF (el mismo que ve el worker)."""
    doc = pymupdf.open(str(ruta))
    try:
        return [(doc.load_page(i).get_text() or "").strip() for i in range(doc.page_count)]
    finally:
        doc.close()


def _argumento(nombre: str, defecto: str | None = None) -> str | None:
    """Valor de ``--nombre valor`` en la línea de comandos."""
    if nombre in sys.argv:
        indice = sys.argv.index(nombre)
        if indice + 1 < len(sys.argv):
            return sys.argv[indice + 1]
    return defecto


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    motor = EngineManager().get_engine("ollama_vision")
    if motor is None:
        print("Ollama Vision no está configurado o habilitado.")
        return 1

    modelo = _argumento("--modelo", MODELO) or MODELO
    ruta = _argumento("--pdf")
    en_euskera = "--eu" in sys.argv
    convenio = "--convenio" in sys.argv
    plantilla = PLANTILLA_EU if en_euskera else PLANTILLA_ES
    if convenio:
        plantilla = PLANTILLA_CONVENIO
        en_euskera = True

    temporal = tempfile.TemporaryDirectory()
    try:
        if ruta:
            pdf = Path(ruta)
            if not pdf.is_file():
                print(f"No existe el archivo: {pdf}")
                return 1
            print(f"documento real: {pdf.name}")
        else:
            pdf = Path(temporal.name) / "documento.pdf"
            paginas = 1 if convenio else PAGINAS
            crear_pdf_sintetico(pdf, paginas, plantilla)
            print(
                "documento sintético: convenio de una página"
                if convenio
                else f"documento sintético: {PAGINAS} páginas"
            )

        textos = texto_de_pdf(pdf)
        completo = "\n\n".join(textos)

        contador = _Contador(motor)
        worker = dw.DocumentWorker(
            contador,
            pdf,
            "resumen",
            modelo,
            scope=dp.SCOPE_GLOBAL if convenio else None,
        )
        inicio = time.perf_counter()
        try:
            resultado = worker.work()
        finally:
            segundos = time.perf_counter() - inicio
    finally:
        temporal.cleanup()

    normalizado = dpl.normalize_pages(textos)
    informe = dpl.check_fidelity(resultado, normalizado.text)
    idioma_documento = worker._plan.language or dp.detect_language(completo)
    idioma_resumen = dp.detect_language(resultado)

    print(f"páginas: {len(textos)} · {len(completo)} caracteres de texto")
    print(
        f"presupuesto MAX_GLOBAL_CHARS = {dp.MAX_GLOBAL_CHARS} → "
        f"entran {len(normalizado.text)} caracteres"
        + ("  ← RECORTADO" if normalizado.truncated else "")
    )
    print(
        f"alcance: {worker._plan.summary} · idioma del plan: {worker._plan.language} "
        f"· llamadas: {contador.texto} texto / {contador.imagen} imagen · {segundos:.1f} s"
    )
    # Con la entrega en dos partes, el «resumen» que se mide es la primera
    # sección (el ejecutivo); el detalle va debajo y no cuenta como resumen.
    ejecutivo = resultado
    if dp.EXECUTIVE_HEADING in resultado:
        _, _, resto = resultado.partition(dp.EXECUTIVE_HEADING)
        ejecutivo = resto.split(dp.DETAIL_HEADING)[0]
    compresion = len(ejecutivo) * 100 / max(len(completo), 1)
    print(
        f"resumen (ejecutivo): {len(ejecutivo)} caracteres, "
        f"{len(ejecutivo.splitlines())} líneas · compresión {compresion:.0f} % "
        f"del documento" + ("   ← NO ESTÁ RESUMIENDO" if compresion > 40 else "")
    )
    if ejecutivo is not resultado:
        print(
            f"detalle por apartados: {len(resultado) - len(ejecutivo)} caracteres "
            f"(entrega total {len(resultado)}, "
            f"{len(resultado) * 100 / max(len(completo), 1):.0f} %)"
        )
    print(f"idioma: documento={idioma_documento} resumen={idioma_resumen}")
    if idioma_sospechoso(resultado):
        print("IDIOMA: el resumen parece estar en inglés (¡revisar!)")
    print(f"cifras inventadas: {len(informe.invented)} {list(informe.invented[:12])}")
    print(f"cifras omitidas: {len(informe.omitted)} {list(informe.omitted[:12])}")
    print("--- resumen completo ---")
    print(resultado)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
