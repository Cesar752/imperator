"""Banco de medida del resumen de documentos (Ollama en crudo, sin Qt).

Sirve para responder con números, no con intuición, a preguntas del estilo
«¿por qué un documento de 19 páginas tarda 20 minutos?».

Mide, con los mismos prompts que usa la app (:mod:`services.document_params`):

- tiempo total de una llamada de resumen **por página** (19 de ellas son el
  camino real del modo Resumen sin la casilla global);
- tokens de prompt y generados, y tamaño del ``thinking`` (los modelos
  qwen3-vl razonan antes de responder y ese razonamiento se come el tiempo);
- el mismo resumen **global** (una sola llamada con el documento unido), que es
  la alternativa de la casilla «Resumen global»;
- el mismo trabajo con un modelo de **texto** en vez del de visión, porque un
  PDF con capa de texto no necesita visión en absoluto.

Uso (con Ollama arrancado y el venv del proyecto)::

    venv\\Scripts\\python.exe tools\\bench_resumen.py

No importa Qt ni el motor de la app: llama a ``/api/chat`` directamente.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services import document_params as dp  # noqa: E402

URL = "http://localhost:11434/api/chat"

#: Una página de texto «corta» como las del caso real: ~1200 caracteres.
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

#: Casos a medir: (descripción, modelo, num_ctx, alcance).
#: Ojo: con ``OLLAMA_NUM_PARALLEL=2`` el contexto se reparte entre dos slots,
#: así que el contexto útil por petición es la mitad del ``num_ctx``.
CASOS = [
    ("vision 8b, 16k (8k/slot), por pagina", "qwen3-vl:8b", 16384, "pagina"),
    ("vision 8b, 32k (16k/slot), por pagina", "qwen3-vl:8b", 32768, "pagina"),
    ("vision 8b, 64k (32k/slot), por pagina", "qwen3-vl:8b", 65536, "pagina"),
    ("texto 9b, 16k, por pagina", "qwen3.5:9b", 16384, "pagina"),
    ("texto 35b, 16k, por pagina", "qwen3.6:35b", 16384, "pagina"),
    ("vision 8b, 32k, GLOBAL (19 paginas)", "qwen3-vl:8b", 32768, "global"),
]

PAGINAS = 19


def llamar(modelo: str, prompt: str, texto: str, num_ctx: int) -> dict:
    """Una llamada a ``/api/chat`` sin streaming y sus métricas."""
    payload = {
        "model": modelo,
        "messages": [{"role": "user", "content": f"{prompt}\n\n{texto}"}],
        "stream": False,
        "options": {"num_ctx": num_ctx},
    }
    peticion = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    inicio = time.perf_counter()
    with urllib.request.urlopen(peticion, timeout=900) as respuesta:
        cuerpo = json.loads(respuesta.read().decode("utf-8"))
    segundos = time.perf_counter() - inicio
    mensaje = cuerpo.get("message", {}) or {}
    return {
        "segundos": segundos,
        "prompt": cuerpo.get("prompt_eval_count", 0),
        "generados": cuerpo.get("eval_count", 0),
        "thinking": len(mensaje.get("thinking") or ""),
        "respuesta": len(mensaje.get("content") or ""),
        "estado": cuerpo.get("done_reason"),
    }


def texto_del_caso(alcance: str) -> str:
    if alcance == "global":
        unido, _ = dp.merge_pages([PAGINA] * PAGINAS)
        return unido
    return PAGINA


def prompt_del_caso(alcance: str) -> str:
    if alcance == "global":
        return dp.global_prompt_for("resumen")
    return dp.text_prompt_for("resumen")


def main() -> int:
    print(f"Pagina de {len(PAGINA)} caracteres; documento simulado de {PAGINAS}.")
    print(
        "Nota: con OLLAMA_NUM_PARALLEL=2 el contexto util por peticion es la "
        "mitad del num_ctx.\n"
    )
    cabecera = (
        f"{'caso':<40} {'s/pag':>7} {'total19':>8} {'prompt':>7} "
        f"{'gen':>6} {'think':>7} {'resp':>6} {'estado':>8}"
    )
    print(cabecera)
    print("-" * len(cabecera))

    for descripcion, modelo, num_ctx, alcance in CASOS:
        texto = texto_del_caso(alcance)
        prompt = prompt_del_caso(alcance)
        try:
            medidas = llamar(modelo, prompt, texto, num_ctx)
        except Exception as exc:  # noqa: BLE001 - el banco no debe morir
            print(f"{descripcion:<40} FALLO: {exc}")
            continue
        total = medidas["segundos"] if alcance == "global" else (
            medidas["segundos"] * PAGINAS
        )
        print(
            f"{descripcion:<40} {medidas['segundos']:>7.1f} {total / 60:>7.1f}m "
            f"{medidas['prompt']:>7} {medidas['generados']:>6} "
            f"{medidas['thinking']:>7} {medidas['respuesta']:>6} "
            f"{str(medidas['estado']):>8}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
