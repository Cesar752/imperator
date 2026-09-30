"""Verificación real del alcance automático de Documentos (con Ollama de verdad).

Crea un PDF de 19 páginas con texto embebido y lo procesa con el motor de visión
configurado en ``config/engines.json``, usando el ``DocumentWorker`` real (no
dobles). Sirve para comprobar, de punta a punta, que un documento largo ya **no**
hace una llamada por página.

Comparativa medida con ``tools/bench_resumen.py`` (misma página de ~900
caracteres, ``qwen3-vl:8b``):

- por página: ~49,9 s por página → 19 páginas ≈ **15,8 min** (19 llamadas);
- global: **47,7 s** para el documento completo (1 llamada).

Uso (Ollama arrancado y el venv del proyecto)::

    venv\\Scripts\\python.exe tools\\verify_alcance_documentos.py

No importa la UI: crea el worker directamente y llama a ``work()``.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# La consola de Windows (cp1252) no traga algunos caracteres de esta salida.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402

from core.engine_manager import EngineManager  # noqa: E402
from ui import document_widget as dw  # noqa: E402

PAGINAS = 19

PAGINA = (
    "ACTA DE LA REUNION ORDINARIA DEL CONSEJO DE ADMINISTRACION\n\n"
    "1. Aprobacion del acta anterior. Se aprueba por unanimidad el acta de la "
    "sesion celebrada el pasado mes, sin modificaciones.\n\n"
    "2. Estado de tesoreria. La tesorera informa de un saldo disponible de "
    "cuarenta y dos mil trescientos euros. Los gastos ordinarios del trimestre "
    "ascienden a dieciocho mil doscientos euros y no hay partidas pendientes de "
    "pago a proveedores.\n\n"
    "3. Vacaciones del personal. Se acuerda mantener el criterio vigente: dos "
    "dias y medio por mes trabajado, con un maximo de treinta dias naturales por "
    "ano. Las solicitudes se presentaran antes del quince de mayo.\n\n"
    "4. Mantenimiento del edificio. Se acepta el presupuesto de la empresa "
    "instaladora por siete mil cuatrocientos euros para la revision anual de la "
    "caldera y el cambio de la bomba de recirculacion.\n\n"
    "5. Ruegos y preguntas. No hay ruegos. Se levanta la sesion a las diecinueve "
    "horas y treinta minutos."
)


class _Contador:
    """Envuelve el motor real para contar cuántas llamadas hace el worker."""

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


def crear_pdf(destino: Path, paginas: int) -> None:
    """PDF con ``paginas`` páginas de texto embebido (sin necesidad de OCR)."""
    doc = pymupdf.open()
    for numero in range(1, paginas + 1):
        page = doc.new_page()
        page.insert_textbox(
            pymupdf.Rect(50, 50, 545, 790),
            f"Pagina {numero}\n\n{PAGINA}",
            fontsize=9,
        )
    doc.save(str(destino))
    doc.close()


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    motor = EngineManager().get_engine("ollama_vision")
    if motor is None:
        print("Ollama Vision no está configurado o habilitado.")
        return 1
    modelo = motor._modelo if hasattr(motor, "_modelo") else "qwen3-vl:8b"

    with tempfile.TemporaryDirectory() as temporal:
        pdf = Path(temporal) / "documento.pdf"
        crear_pdf(pdf, PAGINAS)
        print(f"PDF de {PAGINAS} páginas creado ({pdf.stat().st_size} bytes).")

        contador = _Contador(motor)
        worker = dw.DocumentWorker(contador, pdf, "resumen", modelo)
        inicio = time.perf_counter()
        try:
            resultado = worker.work()
        finally:
            segundos = time.perf_counter() - inicio

    print(f"modelo: {modelo} · alcance resuelto: {worker._plan.summary}")
    print(f"fases: {' → '.join(worker._plan.phases())}")
    print(
        f"llamadas al modelo: {contador.texto} de texto / {contador.imagen} de imagen"
    )
    print(f"tiempo total: {segundos:.1f} s ({segundos / 60:.1f} min)")
    print(f"respuesta: {len(resultado)} caracteres")
    print("--- primeras líneas ---")
    print("\n".join(resultado.splitlines()[:10]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
