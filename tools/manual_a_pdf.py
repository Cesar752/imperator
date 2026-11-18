"""Genera el **Manual de usuario de Imperator** en PDF (sin dependencias nuevas).

El texto del manual vive aquí abajo (constante ``MANUAL``) con un marcado mínimo:

    #  Título de capítulo (H1)
    ## Apartado (H2)
    ### Subapartado (H3)
    -  Viñeta
      -  Viñeta de segundo nivel
    >  Nota / aviso
    |  Tabla (líneas que empiezan por «|» se pintan monoespaciadas)
    (4 espacios) bloque monoespaciado

Se maqueta en A4 con matplotlib (ya presente) y se deja en la raíz del proyecto::

    venv\\Scripts\\python.exe tools\\manual_a_pdf.py

Incluye portada, índice (con números de página) y numeración al pie.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Fuentes TrueType en el PDF: el texto queda como texto real (seleccionable y
# buscable) en vez de como glifos Type 3 que no se pueden extraer.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

#: Dónde se deja el PDF (raíz del proyecto).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DESTINO = PROJECT_ROOT / "MANUAL_IMPERATOR.pdf"

#: A4 en pulgadas y márgenes.
PAGE_W, PAGE_H = 8.27, 11.69
MARGIN_X = 0.6
TOP_Y = PAGE_H - 0.55
BOTTOM_Y = 0.6
USABLE_W = PAGE_W - 2 * MARGIN_X

#: Estilos por tipo de elemento: (tamaño_pt, peso, espacio_antes_in, espacio_despues_in, indentado_in)
ESTILOS = {
    "h1": (17.0, "bold", 0.22, 0.08, 0.0),
    "h2": (12.5, "bold", 0.16, 0.05, 0.0),
    "h3": (10.5, "bold", 0.12, 0.04, 0.0),
    "p": (9.0, "normal", 0.0, 0.06, 0.0),
    "bullet": (9.0, "normal", 0.0, 0.03, 0.24),
    "bullet2": (9.0, "normal", 0.0, 0.03, 0.52),
    # El peso va en "normal": la cursiva se aplica con fontstyle (matplotlib no
    # acepta "italic" como fontweight y el PDF reventaba al pintar una nota).
    "note": (8.6, "normal", 0.0, 0.06, 0.16),
    "mono": (7.9, "normal", 0.0, 0.02, 0.18),
}
FACTOR_ANCHO_MONO = 0.6  # ancho de carácter (monoespaciada) en fracción del tamaño


def _lineas_envoltura(texto: str, tamano: float, indentado: float, prefijo: str) -> list[str]:
    """Parte ``texto`` por palabras al ancho disponible, con sangría colgante."""
    ancho_util = USABLE_W - indentado
    por_linea = max(12, int(ancho_util / (tamano * FACTOR_ANCHO_MONO / 72.0)))
    palabras = texto.split()
    if not palabras:
        return [prefijo.rstrip()]
    salida: list[str] = []
    actual = prefijo
    for palabra in palabras:
        if actual == prefijo:
            actual = actual + palabra
        elif len(actual) + 1 + len(palabra) <= por_linea:
            actual = actual + " " + palabra
        else:
            salida.append(actual.rstrip())
            actual = " " * len(prefijo) + palabra
    if actual.strip():
        salida.append(actual.rstrip())
    return salida or [prefijo.rstrip()]


def _elementos(manual: str) -> list[dict]:
    """Convierte el texto marcado en una lista de elementos tipográficos."""
    elementos: list[dict] = []
    for cruda in manual.splitlines():
        linea = cruda.rstrip()
        if not linea.strip():
            elementos.append({"kind": "espacio"})
            continue
        if linea.startswith("### "):
            elementos.append({"kind": "h3", "text": linea[4:].strip()})
        elif linea.startswith("## "):
            elementos.append({"kind": "h2", "text": linea[3:].strip()})
        elif linea.startswith("# "):
            elementos.append({"kind": "h1", "text": linea[2:].strip()})
        elif linea.startswith("> "):
            elementos.append({"kind": "note", "text": linea[2:].strip()})
        elif linea.startswith("|"):
            elementos.append({"kind": "mono", "text": linea})
        elif linea.startswith("    "):
            elementos.append({"kind": "mono", "text": linea[4:]})
        elif linea.startswith("  - "):
            elementos.append({"kind": "bullet2", "text": linea[4:].strip()})
        elif linea.startswith("- "):
            elementos.append({"kind": "bullet", "text": linea[2:].strip()})
        else:
            elementos.append({"kind": "p", "text": linea.strip()})
    return elementos


def _alto_elemento(elemento: dict) -> float:
    """Alto (pulgadas) que ocupa un elemento ya tipografiado."""
    if elemento["kind"] == "espacio":
        return 0.08
    tamano, _peso, antes, despues, indentado = ESTILOS[elemento["kind"]]
    prefijo = "• " if elemento["kind"] in ("bullet", "bullet2") else ""
    trozos = _lineas_envoltura(elemento["text"], tamano, indentado, prefijo)
    return antes + len(trozos) * (tamano * 1.42 / 72.0) + despues


def _paginar(elementos: list[dict]) -> tuple[list[list[dict]], list[tuple[int, str, int]]]:
    """Reparte los elementos en páginas.

    Devuelve ``(paginas, indice)`` donde ``indice`` son las entradas del índice:
    ``(nivel, titulo, pagina_0based)``.
    """
    paginas: list[list[dict]] = []
    indice: list[tuple[int, str, int]] = []
    actual: list[dict] = []
    y = TOP_Y

    def cerrar() -> None:
        nonlocal actual, y
        if actual:
            paginas.append(actual)
        actual = []
        y = TOP_Y

    for pos, elemento in enumerate(elementos):
        alto = _alto_elemento(elemento)
        siguiente = elementos[pos + 1] if pos + 1 < len(elementos) else None
        # Evita títulos huérfanos: exige que quepa el título + algo después.
        necesidad = alto
        if elemento["kind"] in ("h1", "h2", "h3") and siguiente is not None:
            necesidad += _alto_elemento(siguiente)
        if actual and (y - necesidad) < BOTTOM_Y:
            cerrar()
        if elemento["kind"] in ("h1", "h2") and elemento["text"]:
            indice.append((1 if elemento["kind"] == "h1" else 2, elemento["text"], len(paginas)))
        actual.append(elemento)
        y -= alto
    if actual:
        paginas.append(actual)
    return paginas, indice


def _texto(figura, x_in: float, y_in: float, texto: str, **kw) -> None:
    """Dibuja texto con coordenadas en **pulgadas** (matplotlib usa fracciones)."""
    figura.text(x_in / PAGE_W, y_in / PAGE_H, texto, **kw)


def _pintar_pagina(pdf: PdfPages, elementos: list[dict], numero: int, pie: bool = True) -> None:
    """Dibuja una página con sus elementos y opcionalmente el pie numerado."""
    figura = plt.figure(figsize=(PAGE_W, PAGE_H))
    y = TOP_Y
    for elemento in elementos:
        if elemento["kind"] == "espacio":
            y -= 0.08
            continue
        tamano, peso, antes, _despues, indentado = ESTILOS[elemento["kind"]]
        y -= antes
        color = "#444444" if elemento["kind"] == "note" else "black"
        prefijo = "• " if elemento["kind"] in ("bullet", "bullet2") else ""
        trozos = _lineas_envoltura(elemento["text"], tamano, indentado, prefijo)
        for trozo in trozos:
            _texto(
                figura,
                MARGIN_X + indentado,
                y,
                trozo,
                fontsize=tamano,
                fontweight=peso,
                fontstyle="italic" if elemento["kind"] == "note" else "normal",
                color=color,
                family="monospace",
                va="top",
            )
            y -= tamano * 1.42 / 72.0
    if pie:
        _texto(figura, MARGIN_X, 0.4, "Imperator · Manual de usuario", fontsize=7.5,
               family="monospace", color="#777777")
        _texto(figura, PAGE_W - MARGIN_X, 0.4, f"Página {numero}", fontsize=7.5,
               family="monospace", color="#777777", ha="right")
    pdf.savefig(figura)
    plt.close(figura)


def _pintar_portada(pdf: PdfPages) -> None:
    figura = plt.figure(figsize=(PAGE_W, PAGE_H))
    _texto(figura, PAGE_W / 2, PAGE_H * 0.72, "IMPERATOR", fontsize=34, fontweight="bold",
           family="monospace", ha="center")
    _texto(figura, PAGE_W / 2, PAGE_H * 0.665, "Manual de usuario", fontsize=20,
           family="monospace", ha="center", color="#333333")
    _texto(figura, PAGE_W / 2, PAGE_H * 0.60, "Plataforma modular de IA (Python + PyQt6)",
           fontsize=11, family="monospace", ha="center", color="#666666")
    _texto(figura, PAGE_W / 2, PAGE_H * 0.33,
           "Guía exhaustiva de todas las pestañas y controles:\n"
           "Chat · Documentos · Imagen · Foto · Vídeo · Vídeo IA · Comprensión\n"
           "Cuaderno · Crónica · Audio · Ensamblador · Motores",
           fontsize=10.5, family="monospace", ha="center", va="center", color="#222222")
    _texto(figura, PAGE_W / 2, PAGE_H * 0.12,
           f"Generado el {datetime.now():%d/%m/%Y}", fontsize=9,
           family="monospace", ha="center", color="#777777")
    pdf.savefig(figura)
    plt.close(figura)


def _paginar_indice(
    indice: list[tuple[int, str, int]], offset: int
) -> list[list[tuple[int, str, int]]]:
    """Reparte las entradas del índice en tantas páginas como hagan falta."""
    paginas: list[list[tuple[int, str, int]]] = []
    actual: list[tuple[int, str, int]] = []
    y = TOP_Y - 0.4
    for nivel, titulo, pagina_contenido in indice:
        alto = 0.205 if nivel == 1 else 0.175
        if actual and y - alto < BOTTOM_Y:
            paginas.append(actual)
            actual = []
            y = TOP_Y - 0.4
        actual.append((nivel, titulo, offset + pagina_contenido + 1))
        y -= alto
    if actual:
        paginas.append(actual)
    return paginas or [[]]


def _pintar_pagina_indice(
    pdf: PdfPages, entradas: list[tuple[int, str, int]], numero: int, primero: bool
) -> None:
    """Dibuja una página del índice (con el título «Índice» solo en la primera)."""
    figura = plt.figure(figsize=(PAGE_W, PAGE_H))
    y = TOP_Y
    if primero:
        _texto(figura, MARGIN_X, y, "Índice", fontsize=17, fontweight="bold",
               family="monospace", va="top")
        y -= 0.4
    for nivel, titulo, pagina in entradas:
        x = MARGIN_X + (0.0 if nivel == 1 else 0.28)
        etiqueta = titulo if nivel == 1 else f"· {titulo}"
        _texto(figura, x, y, etiqueta, fontsize=9.2 if nivel == 1 else 8.7,
               fontweight="bold" if nivel == 1 else "normal",
               family="monospace", va="top")
        _texto(figura, PAGE_W - MARGIN_X, y, str(pagina), fontsize=9 if nivel == 1 else 8.7,
               family="monospace", va="top", ha="right")
        y -= 0.205 if nivel == 1 else 0.175
    _texto(figura, MARGIN_X, 0.4, "Imperator · Manual de usuario", fontsize=7.5,
           family="monospace", color="#777777")
    _texto(figura, PAGE_W - MARGIN_X, 0.4, f"Página {numero}", fontsize=7.5,
           family="monospace", color="#777777", ha="right")
    pdf.savefig(figura)
    plt.close(figura)


def escribir_pdf(destino: Path, manual: str) -> Path:
    elementos = _elementos(manual)
    paginas, indice = _paginar(elementos)
    # La portada es la página 1; el índice ocupa las siguientes. El número de
    # páginas de índice se calcula primero (no depende de la numeración) para
    # saber en qué página absoluta cae cada capítulo.
    toc_pages = len(_paginar_indice(indice, offset=0))
    offset = toc_pages + 1  # páginas de contenido -> número absoluto
    paginas_indice = _paginar_indice(indice, offset)
    with PdfPages(destino) as pdf:
        _pintar_portada(pdf)
        for k, pagina_indice in enumerate(paginas_indice):
            _pintar_pagina_indice(pdf, pagina_indice, numero=2 + k, primero=(k == 0))
        for i, pagina in enumerate(paginas):
            _pintar_pagina(pdf, pagina, numero=offset + i + 1)
    return destino


MANUAL = """
# 1. Cómo empezar

## 1.1 Arranque
- Lanza el programa con doble clic en Imperator.bat (abre la ventana sin consola).
- La primera vez aparece un buscador de \"logo\" (splash) y luego la ventana principal.
- El menú superior tiene Archivo, Motores, Ver y Ayuda.

## 1.2 Motores y configuración
- Motores → Gestionar motores... abre la configuración de cada motor (Ollama, ComfyUI, BFL, Piper, Kokoro, Whisper, MusicGen...). Aquí se activan/desactivan y se ajustan rutas, modelos y parámetros.
- Motores → Liberar VRAM al cambiar de pestaña: si está activado, al salir de Chat/Documentos se descarga el modelo de Ollama para dejar sitio a Imagen/Vídeo.
- Motores → Liberar VRAM ahora: suelta la memoria de la GPU al instante (útil si un trabajo se queda sin VRAM).
- Motores → Estado de la GPU...: muestra el estado del planificador (trabajos en curso y memoria libre).
- Archivos de configuración: config/engines.json (motores), config/apis.json (APIs de pago) y config/settings.json (tema, tamaño de letra, limpieza de salidas, planificador de GPU).

## 1.3 Carpetas de trabajo
- output/: resultados temporales (retoques, replays, rótulos, subidas IA...). Se limpia por antigüedad y tamaño.
- projects/: proyectos de vídeo y grabaciones. Incluye projects/grabaciones/ para comprimidos y uniones (NO se borran solos).
- music/: tus MP3 de fondo para el Ensamblador.
- voices/: voces de Piper (modelos ONNX descargados).
- cache/: fotogramas extraídos (se poda sola).

## 1.4 Formatos y elementos comunes
- Vídeo: .mp4 .mkv .mov .avi .webm .m4v .mts .m2ts (los .MTS de cámara AVCHD incluidos).
- Imagen: .png .jpg .jpeg .bmp .webp .tiff .tif (y más al abrir en Foto).
- Todos los trabajos largos muestran una barra de progreso con porcentaje y un reloj (\"Tiempo: m:ss\") y se pueden cancelar.
- Mientras hay un trabajo en curso, los controles se bloquean salvo \"Cancelar\".

## 1.5 El chat como lanzador (router)
- En la pestaña Chat puedes pedir cosas como \"hazme un guion de 4 escenas sobre...\" o \"resume este documento\" y el programa detecta a qué pestaña corresponde.
- Si el mensaje trae lo necesario (p. ej. el guion), abre esa pestaña y arranca la tarea; si no, te pregunta si quieres abrirla.

## 1.6 Ajustes por defecto: máxima calidad
- El programa viene de fábrica con los ajustes de **más calidad** (más fotogramas, más frases, todas las señales); si un trabajo te resulta demasiado lento, baja esos valores en su pestaña.
- Comprensión: un fotograma cada **2 s**, **máx. 120** fotogramas, **Objetos + Etiquetas + OCR + Transcripción** activados.
- Cuaderno: **Visión 120** fotogramas y las tres señales de percepción (incluido OCR).
- Crónica: **60 frases**, un fotograma cada **30 s**, **máx. 120** fotogramas, audio original conservado.
- Imagen: **30 steps**; Foto: mejora IA x4 si marcas la casilla; Documentos: detalle «Ejecutivo + apartados» y alcance «Automático».
- La **visión** usa por defecto `qwen2.5vl:7b` (sin razonamiento, rápido y fiable por fotograma) y el **chat** `qwen3.6:35b` con `think: false`.

## 1.7 Para qué sirve cada pestaña (resumen)
> Si solo quieres saber «dónde hago esto», mira esta tabla; cada capítulo detalla sus controles.

| Pestaña      | Para qué sirve (en una frase)                         | Qué entrega |
| Chat         | Hablar, redactar y LANZAR tareas de otras pestañas    | Texto |
| Documentos   | OCR, resumen y tablas de un PDF o imagen              | Texto (.txt) |
| Imagen       | Crear o editar una imagen desde un prompt             | PNG |
| Retoque      | Arreglar una foto (color) y mejorarla con IA          | Imagen |
| Vídeo        | Crear un vídeo por escenas a partir de un texto       | MP4 |
| Vídeo IA     | Retocar/montar un vídeo que ya tienes                 | MP4 |
| Comprensión  | Ver QUÉ pasa: fotogramas, objetos, texto, audio       | comprension.json |
| Cuaderno     | Analizar el relato y FECHAR cada momento              | informe.json + .md |
| Crónica      | Partido comentado: locución + subtítulos              | MP4 + .md |
| Audio        | Texto a voz (y clonar una voz)                        | WAV/MP3 |
| Ensamblador  | Montaje final: vídeo + voz + música + subtítulos      | MP4 |

# 2. Chat

## 2.1 Qué hace
- Conversación con un modelo de lenguaje local (Ollama) o de una API compatible. Las respuestas se muestran con formato (negritas, tablas, código).

## 2.2 Controles
- Motor: selector del motor de chat disponible (Ollama, OpenAI-compatible, DeepSeek...). Aparece solo si hay más de uno.
- Modelo: modelo a usar; el tooltip muestra su tamaño en disco. Al cambiar de modelo con conversación abierta, se pregunta si limpiar el historial.
- Cuadro de conversación: solo lectura; el autor de cada turno va resaltado (Tú / modelo).
- Barra de escritura: escribe el mensaje y pulsa Enter (o el botón Enviar).
- Indicador \"Pensando...\": muestra el avance mientras el motor responde (streaming).

## 2.3 Notas
- El historial que se envía al modelo se recorta a una ventana reciente (no crece sin fin).
- El contexto (num_ctx) por defecto es 8192 y el número de tokens de respuesta no está limitado; se ajusta por motor en engines.json.
- \"Cancelar\" es implícito: mientras el motor responde no se puede enviar otro mensaje.

## 2.4 Para qué sirve (y cuándo no)
- Para conversar, redactar, traducir, resumir y, sobre todo, como LANZADOR: muchas órdenes abren y arrancan otras pestañas (guion de vídeo, documento, etc.).
- Úsalo para pedir ideas de prompt (imagen) o guiones (crónica), que luego pegas en su pestaña.
- Elegir modelo: uno pequeño responde rápido; uno grande (qwen3.6:35b) razona mejor pero tarda más y ocupa más VRAM.
- No lo uses para tareas de imagen/vídeo: el Chat solo escribe texto.

# 3. Documentos

## 3.1 Qué hace
- Lee un PDF o una imagen y devuelve OCR, un resumen o sus tablas, con el modelo de visión local (Ollama Vision). El trabajo va por fases: detección → OCR/tablas → resumen → validación → entrega.

## 3.2 Controles
- Seleccionar archivo...: PDF o imagen (.png .jpg .jpeg .bmp .webp .tiff .tif). Debajo se muestra el tipo y el tamaño del archivo.
- Modo:
    - OCR: transcribe todo el texto visible respetando el idioma original (no traduce).
    - Resumen: resume el contenido (admite alcance y detalle, ver abajo).
    - Extraer tablas: devuelve las tablas en HTML.
- Modelo: modelo de visión (p. ej. qwen2.5vl:7b, sin razonamiento y rápido). El tooltip avisa de la cobertura de idiomas. Los qwen3-vl ven bien pero «piensan»: por página o por fotograma conviene uno sin razonamiento.
- Idioma: idioma de salida del resumen y las tablas: Auto (respeta el del documento), Español, Euskera, Inglés. El OCR nunca traduce.
- Alcance (solo en Resumen):
    - Automático: resume el documento completo de una vez a partir de 6 páginas; por debajo, página a página.
    - Global: fuerza el resumen del documento entero en una sola llamada.
    - Por página: un resumen por cada página.
- Detalle (solo en Resumen):
    - Ejecutivo + detalle por apartados (por defecto): resumen breve arriba y el detalle por apartado debajo.
    - Ejecutivo (breve): en torno a un quinto del documento, conserva lo que decide algo.
    - Por apartado (detallado): conserva todos los ítems y cifras (ocupa casi lo mismo que el documento).
- Frase de cierre por apartado (→): añade una línea de cierre extractiva a cada apartado. Apagada por defecto (para evitar interpretaciones).
- Forzar OCR (FASE 1): transcribe con visión aunque el PDF ya traiga texto (útil si ese texto es malo).
- Extraer tablas además (FASE 1): añade un bloque de tablas por página junto al resultado principal.
- Procesar / Cancelar; Guardar como .txt; Copiar al portapapeles.

## 3.3 Consejos
- Un PDF largo se resume antes y mucho más rápido con alcance Global que página a página.
- El idioma de salida permite pedir el resumen en euskera aunque el documento venga en castellano (y al revés).

## 3.4 Para qué sirve (y cuándo no)
- Para convertir un documento escaneado o un PDF en algo útil: texto fiel (OCR), resumen (ejecutivo o por apartados) y tablas.
- Úsalo cuando el documento no tenga capa de texto o tenga un texto malo; si el PDF trae texto bueno, deja "Forzar OCR" apagado (ahorra tiempo).
- El resumen respeta el idioma del documento salvo que elijas otro en "Idioma" (el OCR nunca traduce).
- No lo uses para leer el marcador de un vídeo: para eso está el OCR de Comprensión/Cuaderno.

# 4. Imagen

## 4.1 Qué hace
- Genera o edita imágenes: modelos locales por ComfyUI (gratis) o la API de pago Black Forest Labs (BFL).

## 4.2 Controles
- Modelo: lista los modelos disponibles. Los de pago se marcan con \"· PAGO\" y muestran los créditos restantes.
- Créditos: solo con modelos de pago; consulta el saldo de la API en segundo plano.
- Prompt: descripción de lo que quieres.
- Prompt negativo: lo que NO quieres que aparezca (opcional).
- Referencia: imagen de entrada para edición.
    - Obligatoria en flux_kontext (local), flux-kontext-pro y flux-kontext-max (pago).
    - Opcional en flux-2-pro: con foto edita, sin foto crea desde cero.
- Ancho / Alto: 256–2048 px (por defecto 1024×1024).
- Presets: Vertical 9:16 (768×1344), Cuadrado 1:1 (1024×1024), Horizontal 16:9 (1344×768).
- Steps: pasos de muestreo (1–150; **30 por defecto**, calidad).
- Guidance: adherencia al prompt (0–30; 3.5 por defecto).
- Seed: número para reproducir el resultado, o Aleatorio.
- Generar / Cancelar / Guardar como... (PNG). La vista previa aparece debajo.

## 4.3 Para qué sirve (y cuándo no)
- Para crear una imagen desde una descripción (prompt) o EDITAR una foto con instrucciones (con Referencia).
- Local (ComfyUI) es gratis pero depende de tu GPU; BFL es de pago y no necesita GPU potente.
- Con Referencia + flux_kontext mantienes el parecido; con flux-2-pro sin referencia creas desde cero.
- Si solo quieres arreglar color/nitidez de una foto, usa Retoque (no gasta un modelo de imagen).

# 5. Foto (Retoque)

## 5.1 Qué hace
- Abre fotos en cualquier formato, las ajusta con Pillow (siempre disponible) y, si se quiere, las mejora con IA (Real-ESRGAN + GFPGAN) por subprocess.

## 5.2 Controles
- Abrir foto...: muestra las vistas Antes y Después.
- Brillo, Contraste, Color, Nitidez: 0.0–3.0 (1.0 = sin cambio).
- Restablecer ajustes: vuelve a 1.0 en todos.
- Mejora IA: escala x2, x3 o x4 y casilla Restaurar caras (GFPGAN); botón Mejorar con IA.
- Formato: PNG, JPEG, WEBP, TIFF o BMP.
- Calidad: 1–100 (95 por defecto).
- Guardar como...

## 5.3 Notas
- La mejora IA necesita venv-photo (Real-ESRGAN). Si falta, el botón aparece desactivado con la instrucción de instalación.
- La orientación EXIF de las fotos de móvil se respeta al abrir.

## 5.4 Para qué sirve (y cuándo no)
- Para ARREGLAR una foto: ajustes finos e instantáneos (Pillow) y mejora con IA (x2/x3/x4 + restaurar caras).
- Úsala para fotos sueltas; para muchas fotos o para vídeo, usa Vídeo IA (escalado Real-ESRGAN).
- Sin venv-photo, los ajustes de color siguen funcionando; solo se desactiva la mejora IA.

# 6. Vídeo (guion por escenas)

## 6.1 Qué hace
- Genera un guion estructurado con el motor de chat y produce un clip por escena (Wan 2.2), con continuidad visual, y los une en una secuencia.

## 6.2 Generar el guion
- Idea del vídeo: describe de qué va.
- Escenas: número de escenas (1–20; 6 por defecto).
- Generar guion: pide el guion al modelo (Ollama). Mientras llega, la barra muestra los caracteres recibidos.
- Modelo: modelo de Ollama que escribe el guion; botón ↻ para recargar la lista.

## 6.3 La tabla de escenas (editable)
- Columnas: Escena, Narración, Prompt visual y Duración (2–5 s por clip).
- Puedes editar Narración, Prompt visual y Duración. Las filas se pueden reordenar arrastrando (la coherencia del encadenado puede resentirse: se avisa).
- Avisos: un validar ligero marca en amarillo las escenas con problemas (se recalculan al editar).

## 6.4 Continuidad entre escenas
- Transiciones entre escenas: activa el fundido/cruce entre planos. Tipo de transición:
    - (por escena): cada escena usa la suya.
    - Corte, Fundido, Disolvencia, Deslizar (izquierda/derecha), Barrido (izquierda/derecha), Círculo, Zoom.
    - \"Fundido\" = aparición/desvanecido; \"Disolvencia\" = mezcla suave entre las dos imágenes.
- Fotos primero (imagen → vídeo): genera una foto por escena (con su prompt visual) y la usa como primer fotograma del clip i2v. Elige abajo el motor/modelo de imagen. Las fotos quedan en el proyecto y se reutilizan.
- Reanudar proyecto: continúa desde la primera escena que falte en vez de rehacerlo todo. Se conserva la semilla común.

## 6.5 Botones de escenas
- Añadir escena: una escena en blanco al final.
- Ampliar con IA: pide N escenas nuevas (1–10) que continúen la historia, sin reescribir las existentes.
- Generar escenas: genera solo las que falten (o todas).
- Enviar al Ensamblador: pasa la secuencia montada a la pestaña Ensamblador.
- Nuevo: deja la pestaña lista para un vídeo nuevo (no borra el proyecto del disco).
- Borrar proyecto: borra del disco el proyecto cargado y sus clips.
- Proyectos...: abre/borra proyectos guardados.

## 6.6 Menú de una escena (clic derecho)
- Reemplazar clip (MP4)...: pone otro MP4 como clip de esa escena.
- Reemplazar foto (PNG/JPG)...: cambia la foto de arranque (sirve para cambiar un personaje).
- Regenerar esta escena: rehace SOLO esa escena con la foto actual.
- Quitar foto: elimina la foto de la escena.

## 6.7 Parámetros de generación
- Modo: Texto a vídeo (t2v) o Imagen a vídeo (i2v).
- Prompt / Negativo: descripción global (en modo i2v cada foto lleva su propio prompt).
- Ancho / Alto: 256–2048 (832×480 por defecto).
- Duración: 1–30 s; Ajustar al audio... lee un archivo de audio y ajusta la duración.
- FPS: 1–60 (16 por defecto).
- Presets:
    - Horizontal: Rápido 832×480 2 s, Equilibrado 832×480 4 s, Calidad 1024×576 5 s.
    - Vertical (TikTok): Rápido 480×832 2 s, Calidad 720×1280 3 s, Alto 896×1600 4 s (mucha VRAM).
- Seed: fijo o Aleatorio.
- Imagen a vídeo: \"Usar última imagen generada\" y \"Seleccionar imágenes...\"; cada foto añadida tiene su propio prompt.
- Avanzado → Timeout: 300–3600 s (1800 por defecto), el tiempo máximo por clip.

## 6.8 Salida
- Generar / Cancelar / Guardar como... / Abrir carpeta. Los clips y la secuencia se guardan en el proyecto (projects/).

## 6.9 Para qué sirve (y cuándo no)
- Para CREAR vídeo desde un texto: un guion por escenas y un clip por escena con continuidad visual.
- Úsalo para intros, animaciones o escenas inventadas. NO es para comentar un partido (eso es Crónica).
- "Fotos primero" estabiliza a los personajes: genera una foto por escena y el vídeo arranca de ella (i2v).
- Necesita ComfyUI + Wan 2.2 y bastante VRAM: si falla por memoria, baja resolución/duración o usa "Liberar VRAM ahora".

# 7. Vídeo IA (mejora y edición)

## 7.1 Qué hace
- Retoca un vídeo ya existente (color, nitidez, ruido, FPS), cambia su velocidad, crea repeticiones y rampas, quema rótulos, comprime, une grabaciones troceadas y escala con IA.

## 7.2 Elegir vídeo
- Elegir vídeo...: tras elegirlo, se leen duración, resolución, fps y si trae audio.

## 7.3 Color y efectos
- Brillo: −0.5 a +0.5 (0 = sin cambio).
- Contraste: 0 a 3 (1 = sin cambio).
- Saturación: 0 a 3 (1 = sin cambio).
- Gamma: 0.1 a 3 (1 = sin cambio).
- Nitidez: 0 a 2 (0 = sin cambio); enfoca con unsharp.
- Calidez: −100 (frío/azulado) a +100 (cálido/anaranjado). Es la temperatura de color (colortemperature): 0 = neutro (6500 K).
- Volumen: 0–200 % (100 % = igual). Al cambiarlo, el audio se recodifica a AAC.
- Reduce-ruido: off, light, medium o strong (filtro hqdn3d).
- Fotogramas (FPS): Sin cambio, 24, 25, 30, 50 o 60 (interpola con minterpolate).
- Calidad (CRF): 0–35 (18 por defecto; menos = mejor y más pesado).
- Aplicar color/efectos (ffmpeg).

## 7.4 Cámara lenta y repeticiones
- Velocidad: 0.25x a 2.0x (0.5x = mitad de rápido). El audio se reajusta con atempo para no desincronizarse.
- Zoom: 1x–3x; acerca recortando el centro y mantiene la resolución.
- Suavizar el movimiento (interpolar): inventa los fotogramas que faltan antes de estirar el tiempo (solo aplica a la cámara lenta; tarda más). Con vídeo a 50 fps conviene dejarlo DESMARCADO.
- Aplicar velocidad: aplica la velocidad elegida a todo el vídeo.
- Fragmento: \"Del [s] al [s]\", \"· Repeticiones: [1–6]\":
    - Crear repetición: repite el fragmento N veces; con la casilla marcada, la última vuelta va a 0.5x (cámara lenta).
    - Crear rampa (solo el tramo lento): deja el resto a velocidad normal y ralentiza únicamente el fragmento (efecto \"rampa\" de retransmisión).
- Última repetición en cámara lenta: 0.5x.

## 7.5 Rótulos (texto sobre el vídeo)
- Rótulos: una línea por rótulo con el formato \"minuto:segundo frase\". Las líneas que empiezan por # se ignoran. Ejemplo:
    0:00 Balonmano · Jornada 5
    0:12 Gol de Álex · 12-10
    1:05 Parada de Unai
- Duración: 0.5–10 s (lo que dura cada rótulo).
- Posición: Abajo, Centro o Arriba.
- Tamaño: 12–96 px.
- Color: Blanco, Amarillo o Verde.
- Quemar rótulos: incrusta los rótulos en el vídeo (recodifica la imagen; el audio se copia).
- Nota: un rótulo cuyo inicio supera la duración del vídeo avisa en vez de ignorarse.

## 7.6 Comprimir
- Calidad (CRF de CPU / CQ de GPU):
    - Alta (archivo grande): CRF 20 / CQ 26.
    - Equilibrada: CRF 23 / CQ 29.
    - Pequeña (para compartir): CRF 26 / CQ 32.
    - Mínima (móvil y mensajería): CRF 30 / CQ 36.
- Codificador: CPU (libx264, mejor calidad por MB) o GPU (NVENC, más rápido). La opción GPU aparece si tu ffmpeg trae h264_nvenc.
- Comprimir vídeo: recodifica sin tocar la imagen; el audio pasa a AAC 192k. El resultado va a projects/grabaciones/.
- Comprimir varios (lote)...: elige varios ficheros y los comprime seguidos (de uno en uno) con el mismo ajuste, mostrando \"3/10 · 00002.MTS\". Un trozo dañado no tira el lote. Los nombres con el mismo origen no se pisan (se numeran).

## 7.7 Unir grabaciones troceadas (.MTS y similares)
- Lista: Añadir grabaciones... y Quitar seleccionada. Se ordenan por nombre (00000, 00001...), que es el orden de grabación de la cámara.
- Igualar los trozos (recomprimir): déjalo DESMARCADO si todos vienen del mismo proceso (unión rápida copiando streams). Márcalo si alguno está retocado (rampa, cámara lenta, escalado) o no coincide; normaliza tamaño, fps y audio.
- Unir en un solo MP4: deja grabacion_unida_<fecha>.mp4 en projects/grabaciones/.

## 7.8 Escalado con IA y salida
- Escalar IA: x2 o x4 (Real-ESRGAN). Requiere venv-photo.
- Escalar con IA (Real-ESRGAN): extrae fotogramas, los reescala y remonta el vídeo.
- Enviar al Ensamblador: pasa el último vídeo montado (replay, rampa, rótulos, cámara lenta) al Ensamblador.
- Abrir carpeta de resultados: abre en el Explorador la carpeta del último resultado (output/ o projects/grabaciones/).
- Cancelar.

## 7.9 Para qué sirve (y cuándo no)
- Es la "sala de montaje": retoca un vídeo que YA tienes (color, ruido, fps), cambia la velocidad, crea repeticiones y rampas, quema rótulos, comprime y une trozos, y reescala con IA.
- Ideal para dejar listo un partido: comprimir los .MTS, unirlos, sacar el replay del gol y rotularlo.
- Todo va por ffmpeg (rápido; sin GPU salvo NVENC o escalado IA).
- No genera vídeo de la nada: para eso, pestaña Vídeo.

# 8. Comprensión (vídeo)

## 8.1 Qué hace
- Analiza un vídeo y entrega un informe: muestrea fotogramas, los pasa por los motores de percepción (YOLO, CLIP, OCR), transcribe el audio con Whisper y lo resume con Ollama.

## 8.2 Controles
- Elegir vídeo...
- Cada [s]: separación entre fotogramas analizados (1–120 s; **2 por defecto**).
- Máx. fotogramas: tope de fotogramas (1–120; **120 por defecto**, el máximo). Si el vídeo es largo, los fotogramas se reparten uniformemente por todo él.
- Señales: Objetos (YOLO), Etiquetas (CLIP), Texto en pantalla (OCR), Transcribir audio (Whisper).
- Disciplina: Balonmano o Genérico. Sus reglas y vocabulario se le pasan al resumen (sin esto confunde deportes: p. ej. llama «voleibol» a un balonmano). Por defecto, Balonmano. La disciplina es COMPARTIDA con Cuaderno y Crónica: cambiarla aquí la cambia en las demás (y se recuerda al reabrir).
- Idioma: idioma del audio y del resumen (Auto, Español, Euskera, Inglés...). Con «Auto», Whisper detecta solo: si el vídeo tiene mucha grada, puede confundirlo con inglés y el resumen sale lleno de ruido («Come on!»). Fija «Español» cuando el audio sea en castellano.
- Aviso sobre CLIP: las etiquetas son sugerencias. CLIP marca «gol» o «celebración» en muchos fotogramas aunque no pase nada; si el resumen se inventa goles, quita esas etiquetas del vocabulario o desmarca «Etiquetas (CLIP)».
- Etiquetas CLIP: vocabulario (lista separada por comas) que puntúa CLIP. CLIP NO describe: elige entre estas etiquetas, así que con genéricas («calle, coche, animal») da ruido. Por defecto va un vocabulario de balonmano («jugadores en la pista, balón, portería, portero, lanzamiento a portería, gol, siete metros, público en la grada, marcador, árbitro, entrenador, celebración, saludo o fila inicial, tiempo muerto, banquillo, pista cubierta»); edítalo para tu contenido. Recomendado: etiquetas CONCRETAS y separables (acciones y estados del juego), sin nombres ni colores (esos los pone la alineación) y sin dos que signifiquen casi lo mismo.
- Contexto (opcional): datos del partido (fecha, lugar, equipos, alineación, marcador, notas), una línea por dato. Se toman como ciertos y se pasan al resumen para que NOMBRE en vez de describir «un jugador»; además se guardan en comprension.json, así el Cuaderno los adopta (rellena su «Contexto» si lo tienes vacío) y la Crónica los hereda. Ejemplo: «Fecha: 9/10/2026 20:30», «Lugar: Polideportivo de Muskiz», «Alineación PAN MENESA MUSKIZ: 2 ARANE PEREZ…», «Nota: faltan los minutos 30-40».
- Modelo: modelo de chat para el resumen.
- Instrucciones para la IA: qué quieres que destaque el resumen y con qué prioridad (p. ej. goles, paradas y 7 metros). Se pasan al RESUMEN (no a la visión por fotograma); botón \"Poner ejemplo (balonmano)\".
- Analizar / Cancelar / Guardar informe (texto .txt).
- Aviso de disponibilidad: indica qué motores faltan (se omitirán avisando en el informe).

## 8.3 Notas
- Hay que activar al menos una señal.
- El muestreo es por segundos, no por milisegundos: un vídeo a 30 fps serían decenas de inferencias por segundo.
- Al terminar, guarda la percepción estructurada en projects/comprension/<vídeo>_<fecha>/comprension.json: el Cuaderno puede reutilizarla (botón «Usar percepción de Comprensión…») sin repetir YOLO/CLIP/OCR ni Whisper.
- El prompt del resumen se acota (anotaciones ≈ 9.000 caracteres; transcripción ≈ 4.000): la transcripción de un partido de 50 min entera llenaba la ventana de Ollama y salía «se agotó el espacio de la respuesta», incluso con un modelo de resumen pequeño. El informe (y comprension.json) sí guardan la transcripción completa.

## 8.4 Para qué sirve (y cuándo no)
- Para saber QUÉ PASA en un vídeo: anotaciones por fotograma con su minuto, objetos, etiquetas, texto en pantalla y transcripción.
- Es la que "pone los minutos": su comprension.json alimenta al Cuaderno (que verifica) y de ahí a la Crónica.
- Úsala una vez por vídeo; reutilizar su percepción evita repetir YOLO/CLIP/OCR/Whisper.
- No escribe la crónica ni locuta: solo describe y resume.

# 9. Cuaderno (análisis cinematográfico)

## 9.1 Qué hace
- Analiza un vídeo como \"cuaderno de cine\": líneas narrativas, estructura, curva de emoción y segmentos. Todo en local o exportando un paquete ZIP para una IA externa.

## 9.2 Controles
- Elegir vídeo...
- Fotogramas/s: fotogramas que se extraen por segundo (1–10; 1 por defecto).
- Máx. paquete: tope de fotogramas del paquete (1–10800; 1800 por defecto). Si el vídeo supera el tope, los fotogramas se reparten por TODO el vídeo.
- Visión: fotogramas que mira la IA de visión (4–120; **120 por defecto**, el máximo), repartidos por todo el vídeo. Para un partido de ~1 h son uno cada ~26 s; si quieres ir más rápido, bájalo.
- Transcribir audio (Whisper): incluye la transcripción del audio.
- Señales de percepción: Objetos (YOLO), Etiquetas (CLIP) y Texto en pantalla (OCR). Añaden datos estructurados a cada fotograma y quedan en el informe. El OCR viene activado por defecto: es quien lee de forma fiable el marcador y el reloj (la visión ya no los inventa). Requiere tenerlo instalado (pip install paddleocr paddlepaddle; son cientos de MB; el programa ya salva el fallo de oneDNN en CPU).
- Marcador y goles: del texto del OCR se derivan el marcador y los goles con un motor determinista que solo cuenta un gol cuando un lado sube exactamente 1; no inventa resultados ni confunde el reloj con el marcador.
- Etiquetas CLIP: vocabulario (lista separada por comas) que puntúa CLIP; por defecto, uno de balonmano (16 etiquetas concretas y separables, sin nombres ni colores).
- Disciplina: Genérico o Balonmano. Al elegir un deporte se le pasan a la IA sus reglas y su vocabulario (sin esto confunde disciplinas: p. ej. llama «voleibol» a un balonmano). Es COMPARTIDA con Crónica y Comprensión: se sincroniza en vivo entre las tres.
- Contexto: equipos, colores, marcador e incidencias. Se usa al COMPONER el cuaderno (para interpretar y poner nombres); NO se pasa a la visión fotograma a fotograma, porque el «Marcador final» se colaba en casi todas las descripciones (p. ej. «42-29» al minuto 1). El marcador real lo lee el OCR.
- Generar paquete (ZIP)...: extrae los fotogramas y monta un ZIP (fotogramas + instrucciones + esquema JSON) para enviarlo a una IA externa.
- Analizar en local (Ollama): describe los fotogramas con visión y compone el análisis con el chat.
- Importar análisis (JSON)...: carga el JSON devuelto por cualquier IA.
- Exportar Markdown...: guarda el cuaderno en .md.
- Abrir carpeta: abre en el Explorador la carpeta de la entrega (projects/notes/<vídeo>_<fecha>/), donde quedan juntos el ZIP y el analisis.json.
- Usar percepción de Comprensión...: carga el comprension.json que dejó la pestaña Comprensión y reutiliza su percepción por fotograma (objetos/etiquetas/OCR) y su transcripción. Con la casilla «Usar esa percepción» marcada, el análisis NO repite YOLO/CLIP/OCR ni Whisper. Si tu caja «Contexto» está vacía, se rellena con la de la percepción (fecha, lugar, equipos, alineación): así el Cuaderno no la pierde al reutilizarla y el informe.json la lleva a la Crónica.
- Instrucciones para la IA: qué quieres que cuente el cuaderno y en qué orden (p. ej. la alineación al principio y luego goles, paradas y 7 metros). Van al TEXTO del cuaderno; en cada fotograma la visión recibe sólo el mapa color → equipo (LOCAL/VISITANTE). Botón \"Poner ejemplo (balonmano)\".
- Pestañas de resultado: Líneas narrativas, Estructura, Emoción (curva) y Segmentos.

## 9.3 Salida y avisos
- El paquete y el análisis de un mismo vídeo se guardan en la MISMA carpeta de sesión (projects/notes/<vídeo>_<fecha>/). Al elegir otro vídeo se abre una carpeta nueva.
- «Analizar en local» guarda además un informe.json con lo que la visión vio por fotograma (y el audio): la pestaña Crónica puede reutilizarlo para NO volver a analizar el vídeo.
- Flujo recomendado Comprensión → Cuaderno → Crónica: Comprensión analiza el vídeo una vez y deja comprension.json; el Cuaderno lo reutiliza (sin repetir percepción) y deja informe.json; la Crónica usa ese informe sin volver a analizar. Así el análisis y el texto son más exactos sin analizar el vídeo dos veces.
- Marcador y goles: si el OCR está activo, al terminar el análisis verás «Marcador OCR: N cambios leídos, de a-b a c-d; N goles detectados». El informe guarda la serie (marcador) y los goles (goles); la Crónica los recibe como datos reales (no los adivina).
- Contraste de contexto: tras analizar (o al cargar una percepción de Comprensión), compara tu «Contexto» con lo que se ve/oye y avisa SOLO si detecta un choque: marcador que no aparece leído en pantalla (OCR), color de camiseta que no sale en las descripciones/etiquetas, o nombres de equipo que no asoman en la transcripción. Son AVISOS, no correcciones: nada se reescribe solo y tú decides. Para no dar falsos positivos: el marcador se compara tolerando separadores de más («42 --29» = «42 29»); de la línea «Equipos:» se quitan colores y adornos («de azul X. camiseta azul, Y, camisetas rojas» -> «X», «Y»); y los nombres solo se juzgan si la transcripción parece habla en español (con ruido de grada: silbatos, aplausos, inglés suelto, no avisa).
- La pestaña tiene barra de avance y reloj: si el tiempo corre, el trabajo sigue vivo.
- El botón Abrir carpeta se activa en cuanto hay una entrega (paquete o análisis).
- Las anotaciones que recibe la IA para componer el cuaderno se acotan y la transcripción a ≈4.000 caracteres: así caben en la ventana de Ollama y el análisis cubre TODO el vídeo. Si hay MÁS fotogramas que líneas caben (p. ej. 500 con «Visión» alto), se **muestrea** (primera, última y reparto uniforme) en vez de recortar cada descripción a 30 caracteres, que dejaba el bloque ilegible.
- Si aun así el modelo se queda sin espacio, el Cuaderno **reintenta UNA vez** con la mitad de anotaciones (avisa en la barra de estado). El prompt pide además un JSON acotado (máx. 6 líneas narrativas con 12 eventos, 6 segmentos, 12 puntos de emoción y frases de ≤ 12 palabras) para que la respuesta no se corte por tokens.

## 9.4 Para qué sirve (y cuándo no)
- Para analizar el RELATO del vídeo como un cuaderno de cine: líneas narrativas, estructura, curva de emoción y segmentos.
- Es el que "verifica" los minutos: cada t_s sale de una anotación [t=MM:SS]; y guarda informe.json (visión + audio + alineación) que la Crónica reutiliza.
- Úsalo antes de la Crónica si quieres control y detalle, o para exportar el ZIP a una IA externa.
- No locuta ni monta: eso es de Crónica.

# 10. Crónica (partido comentado por IA)

## 10.1 Qué hace
- Convierte un partido en un vídeo comentado: analiza el vídeo, escribe la crónica, la locuta (TTS), añade música opcional y monta el vídeo con subtítulos.

## 10.2 Datos del partido
- Caja \"Datos del partido\": equipos, colores, alineación y marcador (una línea por dato). Lo que escribas se toma como cierto y la crónica usa esos nombres. Los equipos y colores se pasan a la visión por fotograma (para «el jugador azul»), pero el marcador final no: se colaba en todas las descripciones y el marcador real lo lee el OCR.
- Disciplina: Balonmano o Genérico. Sus reglas y vocabulario se le pasan a la IA en la visión por fotograma y en el guion (sin esto confunde deportes: p. ej. llama «voleibol» a un balonmano). Por defecto, Balonmano. Es COMPARTIDA con Cuaderno y Comprensión: se sincroniza en vivo entre las tres.
- Leer acta...: elige el acta (PDF o foto); el programa la lee (texto nativo u OCR) y rellena la caja. Revísala antes de generar.
- Herencia del informe: al cargar un informe del Cuaderno, si la caja está vacía se rellena con los «Datos del partido» del informe (equipos, alineación, colores, marcador). Aunque la dejes vacía, la crónica usa la alineación que venga DENTRO del informe para llamar a las jugadoras por su nombre de pila. Los informes antiguos (sin contexto) no traen alineación: se avisa en la barra de estado y hay que escribirla o re-analizar en el Cuaderno.

## 10.3 Controles
- Idioma: idioma de la crónica.
- Frases: máximo de frases (3–60; **60 por defecto**). Una frase = un momento (se locuta y se subtitula). Un partido de 52 min aguanta 40–60.
- Análisis: \"Un fotograma cada [s]\" (1–60 s; **10 por defecto**) y \"máx. fotogramas\" (4–120; **120 por defecto**); Transcribir el audio (Whisper). Con 30-60 s de salto la visión se pierde goles y paradas: para narrar jugadas, 10 s o menos.
- Instrucciones para la IA: lo que quieres que cuente y en qué orden (p. ej. presentar la ALINEACIÓN al principio, mientras calientan, y luego centrarse en GOLES, PARADAS del portero y 7 METROS). Se pasan al GUION (no a la visión de cada fotograma, donde hacían alucinar alineaciones). El botón \"Poner ejemplo (balonmano)\" rellena un punto de partida; se recuerda entre sesiones.
- Visión: motor que MIRA los fotogramas. Por defecto **Ollama Vision** (local y gratis). Se puede elegir **OpenAI Vision** (GPT-4o, de pago): ve mejor pero cobra por fotograma; la pestaña avisa del gasto. Si cargas el material de Comprensión/Cuaderno, este motor NO se usa (no se analiza el vídeo). Sólo salen motores que SABEN mirar (el modelo debe ser multimodal). El modelo se elige en la lista de al lado: trae los configurados y los que ofrezca el motor (instalados en local o de la API), así se alterna local/pago sin editar engines.json.
- IA que escribe: motor de chat que redacta la crónica. Por defecto, el elegido en la pestaña Chat; aquí puedes usar otro (p. ej. OpenAI o DeepSeek) sólo para la crónica. Ojo: **DeepSeek no tiene visión** (sus modelos son de texto): escribe, pero no mira los fotogramas, así que nunca aparece en «Visión».
- Motor de voz: Piper, Kokoro o Clonar voz (XTTS).
- Voz: voces del motor elegido (cada una muestra su idioma).
- Velocidad de la voz: 0.5–2.0 ×.
- Música: casilla Música de fondo (MusicGen) + descripción (p. ej. \"épico, percusión, pabellón\") + Vol.
- Audio del vídeo: \"Conservar el audio original bajo la voz\" (recomendado para oír el pabellón).
- Volúmenes: Vol. vídeo (ambiente de la grabación; 60 % por defecto) y Vol. voz (narración; 100 %).
- Subtítulos: Tamaño (12–96), Posición (Abajo/Centro/Arriba) y Color (Blanco/Amarillo/Verde).
- Usar informe del Cuaderno…: carga el informe.json que deja la pestaña Cuaderno. Con la casilla «Usar el material cargado» marcada, la crónica se escribe **sin volver a analizar el vídeo** (ni visión ni Whisper): reutiliza lo que el Cuaderno ya vio. Si el informe recuerda el vídeo, se carga solo.
- Usar percepción de Comprensión…: carga el comprension.json que deja la pestaña Comprensión (anotación por fotograma con objetos/etiquetas/OCR, transcripción y resumen). Marcada la casilla, la crónica se escribe **sin volver a mirar el vídeo**: es la vía más rápida y no gasta visión (con una IA de pago, no cuesta por fotograma). Flujo recomendado: Comprensión → Cuaderno → Crónica.
- Generar crónica / Enviar al Ensamblador / Abrir carpeta del resultado / Cancelar.
- Tamaño del informe: lo que se le pasa al modelo se acota (transcripción ≈ 4.000 caracteres; informe ≈ 9.000) para que quepa en la ventana de Ollama. Si aparece «se agotó el espacio de la respuesta», sube num_ctx del motor ollama en engines.json o usa un modelo sin razonamiento.

## 10.4 Salida
- Un MP4 con narración y subtítulos quemados + un <nombre>_cronica.md con el texto y sus marcas de tiempo (hoja de cortes).

## 10.5 Receta para un partido
- 1) Leer acta (o escribir los datos) → 2) Elegir el vídeo final → 3) Ajustes: 40–60 frases, un fotograma cada 10 s y 120 fotogramas para 52 min → 4) Voz de España (es_ES-davefx-medium) → 5) Dejar marcado \"Conservar el audio original\" → 6) Generar crónica.

## 10.6 Calidad de la crónica (juego, goleador, tiempos y voz)
- Apertura: la crónica empieza SIEMPRE con la sede (polideportivo/lugar), la fecha y la ALINEACIÓN DE LOS DOS EQUIPOS. Se compone con reglas desde «Datos del partido» (no la escribe el modelo, así no se olvida). Las frases de la apertura se encadenan por la DURACIÓN REAL del audio ya sintetizado (antes la alineación de un equipo se pisaba con la del otro).
- Momento de cada frase: se queda en el instante del informe; sólo se corrige una deriva de ±3 s. Antes se movía al instante libre más cercano y una frase del calentamiento acababa en el minuto 3:40.
- Local/visitante y marcador: de «Equipos: … (local) vs … (visitante)» se sabe quién es cada uno (izquierda/derecha del marcador) y el marcador real del OCR se etiqueta con los nombres; el guion NO puede decir que va por delante quien no lo esté en esa serie.
- Ataque/defensa: la visión dice qué equipo tiene el balón (ese ataca; el otro defiende) y el guion no puede decir que ataca quien no lo lleva.
- Goles y paradas: el guion dedica UNA frase a cada anotación con GOL, PARADA o 7 METROS del informe.
- Juego vs. descanso: el Cuaderno pide a la visión que cada fotograma EMPIECE por su estado (JUGANDO / SAQUE / GOL / CELEBRACIÓN / TIEMPO MUERTO / DESCANSO / PREVIO / FINAL); el análisis y la crónica solo narran momentos de JUEGO.
- Quién marca: la visión LEE el dorsal (y el marcador) si se ve. Para que el relato llame a cada jugadora por su NOMBRE DE PILA, da la alineación (dorsal + nombre) en «Contexto» (Cuaderno) o «Datos del partido» (Crónica): el programa cruza el dorsal con la alineación y sustituye «dorsal 7» / «la del rojo» por el nombre (vale el nombre completo del acta: se usa el de pila). Si no consta el dorsal, la crónica NO inventa nombres.
- A destiempo: el modelo debe devolver en cada frase su «ref» (el [MM:SS] exacto del informe que comenta) y la app ANCLA la frase a ese instante anotado; si dos frases apuntan al mismo, la segunda pasa al siguiente instante libre (no se pierde ninguna). Así la crónica cae en la misma rejilla que Comprensión y Cuaderno.
- Pocas frases y todas al principio: el INFORME se muestrea para cubrir TODO el vídeo (si hay más fotogramas que líneas caben, se reparten de la primera a la última anotación; antes se recortaba por el final y la crónica solo narraba los primeros minutos). Si aun así salen pocas frases, sube «Frases» y «Máx. fotogramas».
- Identidad de la jugadora: el programa usa la alineación (dorsal + nombre) para poner nombres y el prompt pide mantener el nombre cuando la misma jugadora reaparece sin que se le lea el dorsal. Si no se puede identificar, la crónica dice «una jugadora del equipo azul/rojo» en vez de inventar un nombre.
- Voz: usa Piper con una voz de ESPAÑA (es_ES-…). Evita es_AR (argentina) y Kokoro (su español no es castellano). Velocidad 1.0; Vol. voz 100 % y Vol. vídeo 60 %.

## 10.7 Para qué sirve (y cuándo no)
- Para convertir un partido en un vídeo COMENTADO y LOCUTADO, con subtítulos, de principio a fin y en local.
- Es la que "coloca cada frase donde pasa el hecho": el modelo cita el [MM:SS] y la app ancla la frase, la voz y el subtítulo a ese segundo.
- Reutiliza el informe del Cuaderno: si lo cargas, no vuelve a mirar el vídeo (ni visión ni Whisper).
- Necesita un motor de voz (Piper/Kokoro/XTTS) y ffmpeg. No es para editar color/velocidad (eso, Vídeo IA).

# 11. Audio (texto a voz)

## 11.1 Controles
- Texto: lo que quieres convertir a voz.
- Motor: Piper, Kokoro o Clonar voz (XTTS).
- Voz: voces del motor (con su idioma y, en Piper, de qué país es). Las de España salen primero.
- Velocidad: 0.5–2.0 ×.
- Generar voz / Cancelar / Reproducir / Abrir con el reproductor del sistema / Guardar como... (WAV o MP3).

## 11.2 Notas
- Kokoro no cubre el euskera: para voces en euskera usa Piper.
- \"Clonar voz\" necesita una muestra de audio en voices/clones/.

## 11.3 Para qué sirve (y cuándo no)
- Para pasar CUALQUIER texto a voz y guardarlo (WAV/MP3), o clonar una voz concreta (XTTS).
- Útil para doblar, locutar avisos, generar audiolibros o preparar una voz para el Ensamblador.
- Piper es rápido y ligero (cubre euskera); Kokoro suena natural en inglés; XTTS clona una voz con una muestra.
- No mezcla con vídeo: eso lo hace el Ensamblador o la Crónica.

# 12. Ensamblador (montaje final)

## 12.1 Qué hace
- Combina un MP4 con un audio (voz), subtítulos (Whisper) y una música de fondo, y deja el MP4 final.

## 12.2 Controles
- Vídeo (MP4): Seleccionar vídeo... Muestra resolución, duración y si trae audio (\"Audio: sí/no\").
- Audio (opcional): Seleccionar audio... Muestra su duración. Si difiere mucho del vídeo, se avisa (el vídeo manda).
- Música de fondo:
    - Pista: elige entre los archivos de music/ (botón Refrescar).
    - Vol. música (0–100 %), Vol. voz (0–100 %), Vol. vídeo (0–100 %).
    - Inicio voz (s) / Inicio música (s): cuándo empiezan a sonar (0–30 s).
    - Ducking automático: baja la música mientras hay voz.
    - Conservar el audio del vídeo: mezcla el sonido original (público, silbato) con la voz en vez de sustituirlo.
- Subtítulos: Generar subtítulos con Whisper; Modelo Whisper; Idioma (Auto, Español, Inglés, Euskera, Francés, Alemán, Italiano, Portugués); Estilo → Tamaño (12–72), Posición (Abajo/Centro/Arriba) y Color (Blanco/Amarillo/Verde).
- Ensamblar / Cancelar / Guardar como... / Abrir carpeta.

## 12.3 Notas
- Al recibir una crónica (que ya trae audio) el Ensamblador lo detecta y marca \"Conservar el audio del vídeo\".
- Si el vídeo no tiene audio, esa casilla se desmarca sola (intentar conservarlo haría fallar el montaje).

## 12.4 Para qué sirve (y cuándo no)
- Para el MONTAJE FINAL: un vídeo + una voz + (opcional) música + subtítulos -> un MP4 listo.
- Es el destino de \"Enviar al Ensamblador\" de Vídeo, Vídeo IA y Crónica.
- Con \"Conservar el audio del vídeo\" mantiene el ambiente (público) bajo la voz; el ducking baja la música cuando habla la voz.
- No añade rótulos ni repeticiones: eso ya lo traen Vídeo IA o Crónica.

# 13. Motores y configuración avanzada

- Gestionar motores...: alta/baja y ajuste de cada motor (rutas, modelos, parámetros, APIs).
- Liberar VRAM al cambiar de pestaña y Liberar VRAM ahora.
- Estado de la GPU...: trabajos en curso y memoria libre (planificador de GPU).
- Ver → Modo oscuro, Tamaño de texto (10–14 pt).
- settings.json: log_level, theme, font_size, output_limpieza (retención de output/), gpu_scheduler (max_concurrent, vram_headroom_mb...).
> El detalle motor a motor (para qué sirve cada uno y qué parámetro tocar) está en el capítulo 17; la configuración de los JSON, en el 19.

# 14. Recetario (casos concretos)

## 14.1 Hacer un fundido entre planos (Vídeo)
- Pestaña Vídeo → marca Transiciones entre escenas.
- Tipo de transición: elige Fundido (o Disolvencia).
- Genera las escenas; el cruce se aplica al unir la secuencia.

## 14.2 Cámara lenta de un momento (Vídeo IA)
- Elige el vídeo. En Cámara lenta y repeticiones: pon Velocidad a 0.5x.
- Si quieres que sea suave, marca Suavizar; con 50 fps de origen, déjalo desmarcado.
- Pulsa Aplicar velocidad.

## 14.3 Repetición de un gol (Vídeo IA)
- Elige el vídeo. En Fragmento pon los segundos Del y Al del gol.
- Repeticiones: 3; marca Última repetición en cámara lenta.
- Pulsa Crear repetición.

## 14.4 Rampa de un momento (Vídeo IA)
- Elige el vídeo. Pon Del/Al del momento.
- Pulsa Crear rampa (solo el tramo lento): el resto queda normal.

## 14.5 Ganar nitidez (Vídeo IA)
- En Nitidez sube de 0 hacia 1.0-1.5 (sin pasarse). Si hay mucho ruido, sube Reduce-ruido.
- Aplicar color/efectos (ffmpeg).

## 14.6 Rotular un vídeo (Vídeo IA)
- Escribe los rótulos (uno por línea, \"minuto:segundo frase\").
- Ajusta Duración, Posición, Tamaño y Color, y pulsa Quemar rótulos.

## 14.7 Comprimir todos los trozos de una grabación (Vídeo IA)
- Comprimir varios (lote)... → selecciona los .MTS → los comprime seguidos.
- Los comprimidos quedan en projects/grabaciones/ (bótón \"Abrir carpeta de resultados\").

## 14.8 Unir los trozos en un solo MP4 (Vídeo IA)
- Añadir grabaciones... con los trozos; deja \"Igualar los trozos\" desmarcado si vienen del mismo proceso.
- Unir en un solo MP4.

## 14.9 Crónica de un partido (Crónica)
- Leer acta → revisar Datos del partido → elegir vídeo → 40–60 frases, un fotograma cada 60 s, 60 fotogramas → voz de España → dejar \"Conservar audio original\" → Generar crónica.

# 15. Solución de problemas

## 15.1 No aparecen los .MTS en el selector
- Elige \"Todos (*)\" en el desplegable de tipos de archivo.

## 15.2 No aparece el codificador GPU (NVENC)
- Drivers NVIDIA antiguos o ffmpeg sin NVENC. Con CPU funciona igual, más lento.

## 15.3 Un trozo no baja de peso
- Es material con mucha acción a 50 fps: usa \"Pequeña\" o \"Mínima\".

## 15.4 La unión falla
- Marca \"Igualar los trozos\": un trozo retocado cambia el ritmo y el códec de audio.

## 15.5 El acta no se lee
- Si es una foto, que esté recta y con buena luz. Si el formato es raro, escribe los datos a mano.

## 15.6 El vídeo con voz se queda \"trabajando\" mucho
- Es normal en ffmpeg con relleno de silencio; revisa que tengas la versión actual de Imperator.

## 15.7 La pestaña no deja ver todos los controles
- Las pestañas largas (Vídeo IA, Crónica) tienen barra de desplazamiento vertical: baja con la rueda o la barra para ver todo.

## 15.8 Error de Ollama: «se agotó el espacio de la respuesta (num_predict=-1) o de la ventana de contexto»
- Significa que el prompt no cabía en num_ctx o que el modelo se gastó la ventana «pensando» (qwen3.x razona mucho) antes de responder. En la Crónica ya no debería pasar: el informe se acota (transcripción y total) para dejar sitio a la respuesta.
- Lo primero: apaga el razonamiento del motor de chat con `"think": false` en los `parametros` del motor `ollama` de config/engines.json (ya viene así). Con eso el modelo no gasta la ventana «pensando» y la respuesta cabe.
- Si aún aparece: sube `num_ctx` del motor ollama (p. ej. 32768), usa un modelo sin razonamiento o baja el número de frases/fotogramas.

## 15.9 La crónica habla a destiempo, no distingue juego de saludo, no sabe quién marca o la voz no sale bien
- Juego/saludo: relanza «Analizar en local» con la versión nueva (la visión ahora marca el estado de cada fotograma).
- Goleador: añade la alineación (dorsal + nombre) en «Contexto»; el programa traduce el dorsal al nombre de pila (los prompts lo piden y, además, se reescribe «dorsal 7» → nombre en el texto). Sin alineación no pone nombres: la visión solo lee el dorsal si se ve en el fotograma.
- Error de visión en el Cuaderno («se agotó la ventana de contexto… piensan mucho más de lo que responden»): el motor de visión está usando un modelo con razonamiento (qwen3-vl) que gasta la ventana en cada fotograma. En Motores → Ollama Vision elige `qwen2.5vl:7b` (sin razonamiento); si quieres un qwen3-vl, quita `num_predict` y sube `num_ctx`.
- A destiempo: se ancla cada frase al [MM:SS] anotado («ref»); con más «Visión» (100-120) hay más instantes y cae más cerca de la acción real. Cuaderno y Crónica comparten esa rejilla de minutos.
- Voz: elige Piper y una voz es_ES (castellano); revisa velocidad 1.0 y Vol. voz 100 % / Vol. vídeo 60 %.

# 16. Atajos y trucos

- Enter envía el mensaje en Chat.
- Clic derecho sobre una escena (pestaña Vídeo) abre su menú (reemplazar clip/foto, regenerar...).
- Los ajustes de cada pestaña se recuerdan entre sesiones (settings.json).
- \"Enviar al Ensamblador\" aparece en Vídeo, Vídeo IA y Crónica para encadenar tareas.
- Barra de progreso + reloj: si el tiempo avanza, el trabajo sigue vivo aunque la barra no se mueva.

# 17. Motores y modelos (para qué sirve cada uno)

## 17.1 Qué es un motor y un modelo
- Un MOTOR es la pieza que hace el trabajo (Ollama, ComfyUI, Piper, ffmpeg...). Vive en config/engines.json y se activa/ajusta en «Motores → Gestionar motores...».
- Un MODELO es el cerebro concreto que usa un motor (por ejemplo, Ollama con qwen3.6:35b, o Ollama Vision con qwen3-vl:30b-a3b).
- Si un motor no está instalado o está desactivado, la pestaña que lo necesita se desactiva o avisa; el programa arranca igual.

## 17.2 Chat y visión (Ollama)
- ollama (chat): escribe texto. Lo usan el Chat, los guiones de Vídeo, el análisis del Cuaderno y la crónica. Modelo por defecto qwen3.6:35b (razona; con think=false va más rápido y no se queda sin ventana).
- ollama_vision (visión): mira imágenes. Lo usan Documentos (OCR/resumen) y las descripciones por fotograma de Comprensión, Cuaderno y Crónica. Modelo por defecto qwen2.5vl:7b: es **sin razonamiento**, y por fotograma es lo que funciona (rápido y sin gastar la ventana). Los qwen3-vl ven mejor pero «piensan»: si los usas para vídeo, quita num_predict y sube num_ctx, o el Cuaderno dará «se agotó la ventana de contexto».
- OpenAI / DeepSeek / llama.cpp / LM Studio: chat por API compatible con OpenAI (nube o servidor local). Se eligen en el selector «Motor» del Chat y los respetan las demás pestañas. Son de TEXTO: ni DeepSeek ni llama.cpp/LM Studio ven imágenes (para eso, un motor de visión). En servidores LOCALES (llama.cpp/llama-server, LM Studio) la app consulta GET /models y añade a los selectores los modelos que sirve el servidor (el .gguf cargado), sin editar engines.json.
- openai_vision (visión de pago): mira imágenes por API compatible con OpenAI (GPT-4o, Gemini por su endpoint compatible, xAI…). Usa la MISMA clave que el chat de OpenAI (OPENAI_API_KEY en .env) y se elige sobre todo en la pestaña Crónica (Visión), donde cobra por fotograma; 'detail' y 'max_tokens' acotan el coste.

## 17.3 Imagen y vídeo
- comfyui: genera imágenes con workflows locales (flux_dev, flux_schnell, z_image_turbo, flux_kontext). Gratis; usa tu GPU.
- bfl: API de pago (flux-kontext-pro/max, flux-2-pro). Edita o crea imágenes sin GPU potente; consume créditos.
- wanvideo: vídeo con Wan 2.2 sobre ComfyUI (workflows t2v e i2v). Es el motor de la pestaña Vídeo.
- openai_image: imagen externa (por ejemplo gpt-image-1) como motor de la pestaña Imagen.

## 17.4 Voz, audio y música
- piper_tts: texto a voz local, rápido y ligero; cubre castellano y euskera (busca voces es_ES-...).
- kokoro_tts: voz natural (mejor en inglés; su español no es castellano).
- voice_clone: clona una voz con XTTS-v2 desde una muestra en voices/clones/ (usa venv-tts).
- whisper_engine: transcribe audio a texto (subtítulos del Ensamblador; audio de Comprensión, Cuaderno y Crónica).
- musicgen: genera música de fondo para la Crónica o el Ensamblador.

## 17.5 Percepción (datos duros)
- yolo: detecta objetos (persona, balón...) y sus cajas; el informe resume qué se ve.
- clip: puntúa un vocabulario cerrado que tú eliges (Etiquetas CLIP); no describe.
- paddle_ocr: lee el TEXTO en pantalla (marcador, reloj). Es la fuente fiable del marcador.
- cli_tool (programas externos): Imperator puede invocar ejecutables (bkingfilm, lapian_notes) como motores; aparecen deshabilitados si no están.

## 17.6 Qué parámetro tocar (lo que de verdad importa)
- model: el modelo a usar (por defecto en engines.json).
- num_ctx: la ventana de contexto en tokens. Súbela si un prompt largo no cabe (error «se agotó el espacio...»).
- think: false apaga el razonamiento de los qwen3 (evita perder la ventana pensando); la visión admite además num_predict para acotar la salida.
- timeout: segundos máximos por petición (súbelo con modelos que razonan o vídeo largo).
- ruta/URL: dónde escucha el motor (127.0.0.1 en local). Las CLAVES de APIs se leen del .env.

# 18. Flujos completos paso a paso

## 18.1 Un partido, de la cámara al vídeo comentado
- 1) Vídeo IA → «Comprimir varios (lote)...» con los .MTS → «Unir en un solo MP4» (queda en projects/grabaciones/).
- 2) Crónica → «Leer acta...» (o escríbela) y revisa «Datos del partido» (equipos, colores, alineación, marcador).
- 3) Comprensión → disciplina Balonmano, un fotograma cada 5 s, máx. 60, señales: Objetos + Etiquetas + Texto en pantalla + Transcribir → «Analizar». Deja comprension.json.
- 4) Cuaderno → «Usar percepción de Comprensión...», Visión 60, OCR marcado, el mismo Contexto (con la alineación) → «Analizar en local». Deja analisis.json + informe.json.
- 5) Crónica → «Cargar informe del Cuaderno», 40-60 frases, voz es_ES, «Conservar el audio original» → «Generar crónica». Cada frase cae en su [MM:SS].
- 6) Opcional: Vídeo IA → «Crear repetición» del gol (Del/Al + Repeticiones 3 + última en cámara lenta) y «Quemar rótulos».
- 7) Ensamblador → «Enviar al Ensamblador» → música opcional + subtítulos → «Ensamblar».

## 18.2 Un documento (acta, convenio, informe)
- Documentos → Seleccionar archivo → Modo (OCR o Resumen) → Idioma → en Resumen, Alcance (Automático) y Detalle → Procesar → Guardar como .txt.

## 18.3 Una imagen
- Imagen → elige Modelo (local o de pago) → escribe el Prompt (y el Negativo) → tamaño/presets → Generar → Guardar como.

## 18.4 Un vídeo corto desde texto
- Vídeo → escribe la Idea y las Escenas → «Generar guion» → revisa la tabla → (opcional) «Fotos primero» → «Generar escenas» → «Enviar al Ensamblador».

## 18.5 Arreglar una foto
- Retoque → Abrir foto → ajusta Brillo/Contraste/Color/Nitidez → (opcional) Mejora IA x2/x3/x4 → formato y calidad → Guardar como.

## 18.6 Pasar un texto a voz
- Audio → pega el Texto → Motor y Voz → Velocidad → «Generar voz» → Reproducir o Guardar como (WAV/MP3).

# 19. Configuración a fondo (JSON)

## 19.1 Dónde está cada cosa
- config/engines.json: motores (activar/desactivar, rutas, modelos, parámetros). También se edita desde «Motores».
- config/apis.json: APIs de pago (endpoints). Las CLAVES van en .env (OPENAI_API_KEY, DEEPSEEK_API_KEY...).
- config/settings.json: tema, tamaño de letra, retención de output/, planificador de GPU, última voz/idioma, etc.
- .env: variables de entorno (claves). No se comparte.

## 19.2 Planificador de GPU
- Antes de un trabajo pesado, mide la VRAM libre (nvidia-smi) y decide si cabe; si no, avisa o espera.
- «Motores → Liberar VRAM ahora» descarga el modelo en uso; «Liberar VRAM al cambiar de pestaña» lo hace solo al salir de Chat/Documentos.
- Ajustes en settings.json: gpu_scheduler (max_concurrent, vram_headroom_mb...).

## 19.3 Limpieza de output/
- output/ se limpia por antigüedad (días), tamaño (MB) y número de ficheros, dejando siempre lo más reciente.
- projects/ (proyectos, grabaciones, análisis) NO se borra solo: ahí están tus trabajos.

# 20. Glosario

- Motor: programa que ejecuta una tarea (Ollama, ComfyUI, Piper, ffmpeg...).
- Modelo: red concreta que usa un motor (qwen3, qwen3-vl, flux, piper...).
- Prompt: el texto con el que pides algo a un modelo.
- Prompt negativo: lo que NO quieres que aparezca.
- num_ctx: tamaño de la ventana de contexto (tokens) que ve el modelo.
- think: si el modelo razona antes de responder. false = directo (más rápido y no agota la ventana).
- t2v / i2v: texto-a-vídeo / imagen-a-vídeo.
- FPS: fotogramas por segundo (fluidez).
- CRF / CQ: cuánto comprime el vídeo (menos = mejor calidad y más peso).
- Ducking: bajar la música automáticamente mientras suena la voz.
- OCR: leer texto de una imagen (marcador en pantalla, acta escaneada).
- Percepción: datos estructurados por fotograma (objetos, etiquetas, texto, audio).
- Alineación (roster): lista «dorsal + nombre» de cada equipo; convierte dorsal 7 en Marc.
- ref [MM:SS]: el minuto:segundo de la anotación en la que se basa una frase de la crónica; ancla la frase a ese instante.
- informe.json: lo que el Cuaderno vio (fotogramas + audio + datos del partido); la Crónica lo reutiliza.
- comprension.json: la percepción que deja Comprensión, reutilizable por el Cuaderno.

# 21. Preguntas frecuentes

## 21.1 ¿Necesito internet?
- No para lo local (Ollama, ffmpeg, Piper...). Solo las APIs de pago (BFL, OpenAI, DeepSeek) usan internet.

## 21.2 ¿Se sube mi vídeo o mis documentos a algún sitio?
- No. El análisis y el montaje son locales; solo salen datos si usas un motor de pago.

## 21.3 ¿Por qué una pestaña está desactivada o avisa?
- Porque falta su motor o no está instalado. Abre «Motores → Gestionar motores...» para ver el detalle.

## 21.4 Va lento o se queda sin memoria
- Baja resolución/fotogramas o duración; usa «Liberar VRAM ahora»; cierra otros programas que usen la GPU.

## 21.5 ¿Dónde están mis resultados?
- output/ (temporales), projects/ (proyectos y grabaciones). Cada pestaña tiene «Abrir carpeta».

## 21.6 ¿Puedo parar un trabajo?
- Sí: «Cancelar». Los trabajos con ffmpeg/ComfyUI se interrumpen al momento; otros, al terminar la petición en curso.

# 22. Índice rápido: «quiero hacer...» -> pestaña

- Comentar un partido con voz -> Crónica (o Cuaderno + Crónica para nombres y minutos exactos).
- Saber qué pasa en un vídeo (con minutos) -> Comprensión.
- Analizar el relato (líneas, emoción) -> Cuaderno.
- Leer un PDF o foto y resumirlo -> Documentos.
- Crear una imagen -> Imagen; arreglar una foto -> Retoque.
- Crear un vídeo de la nada -> Vídeo (guion por escenas).
- Comprimir, unir, rotular o repetir un vídeo -> Vídeo IA.
- Pasar texto a voz (o clonar una voz) -> Audio.
- Montar vídeo + voz + música + subtítulos -> Ensamblador.
- Configurar o depurar motores/modelos -> Motores → Gestionar motores...
"""


def main() -> int:
    escribir_pdf(DESTINO, MANUAL)
    print(f"pdf: {DESTINO} ({DESTINO.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
