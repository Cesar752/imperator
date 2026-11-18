"""Crónica locutada de un vídeo: guion con tiempos, voz, música y subtítulos.

Aquí vive todo lo decidible sin Qt ni subprocess: el prompt que pide la crónica al
chat, el parseo de su respuesta (JSON con tiempos o líneas «minuto:segundo frase»),
el SRT que se quema y el comando de ffmpeg que coloca cada frase de voz en su
segundo y las mezcla en una sola pista.

El flujo completo (lo ejecuta ``ui.chronicle_widget``) es:

1. **Análisis**: los fotogramas se describen con visión local y, si se quiere, se
   transcribe el audio con Whisper. Es el informe de Comprensión.
2. **Guion**: el chat escribe la crónica **con marcas de tiempo** desde el informe.
3. **Voz**: cada frase se sintetiza (Piper/Kokoro/XTTS) y se coloca en su segundo.
4. **Música**: opcional, con MusicGen; el Ensamblador la mezcla con *ducking*.
5. **Montaje**: vídeo + voz + música + subtítulos → MP4 final.

Por qué con tiempos: una crónica que se locuta tiene que decir *cuándo* pasa cada
cosa. Con los tiempos se generan de una vez la voz (``adelay`` por frase) y los
subtítulos (``build_srt``), sin pasar el texto por Whisper — que sólo se usa para
entender el audio original, no para subtitular texto que ya conocemos.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from services import notes_params as nts
from services import roster_params as rp
from services import sport_params as sp
from services import video_enhance_params as vep
from services.video_understanding import (
    build_srt,
    format_timestamp,
    parse_timestamp,
    understanding_frames,
)

#: Idioma por defecto de la crónica.
DEFAULT_LANGUAGE = "es"
#: Frases que se le piden al modelo por defecto. Por defecto se apunta a la
#: **máxima calidad**: el tope (60), que para un partido de ~1 h es una frase por
#: minuto. Si la crónica sale larga, se baja desde la pestaña.
DEFAULT_MAX_LINES = 60
#: Tope duro de frases. Un partido de 50-60 min aguanta 40-60 momentos sin hacerse
#: pesado (una frase cada ~1 min); con clips cortos sobran pocas. El límite real lo
#: pone el informe: la IA sólo puede contar lo que ha visto (los fotogramas
#: analizados), así que si se piden más frases que momentos hay que subir también
#: «Un fotograma cada… · Máx. fotogramas», o rellenará con generalidades.
MAX_LINES = 60
#: Longitud admitida de cada frase (recorta respuestas que se van de madre).
MIN_LINE_CHARS = 8
MAX_LINE_CHARS = 240
#: Duración de cada subtítulo: entre estos dos límites, calculada por longitud.
MIN_LINE_S = 1.5
MAX_LINE_S = 8.0
#: Ritmo de lectura con el que se estima lo que dura cada subtítulo.
CHARS_PER_SECOND = 14.0
#: Silencio mínimo entre dos frases: evita que se pisen al mezclar la voz.
MIN_GAP_S = 0.3
#: Retraso máximo admisible al encajar una frase. Si para no montarse con la
#: anterior habría que retrasarla más de esto, el hecho ya pasó y se SALTA (se
#: pasa a la siguiente). Con más eventos detectados (goles/paradas) evita que dos
#: voces se monten o que una llegue a destiempo.
MAX_NARRATION_DELAY_S = 12.0
#: Volumen del audio del vídeo bajo la voz (fracción). 0,6 deja el pabellón de fondo
#: sin tapar al locutor; a 1,0 el ambiente se oye igual de fuerte que la narración.
DEFAULT_ORIGINAL_VOLUME = 0.6
#: Frecuencia a la que se mezcla la narración (Piper y Kokoro traen otra cada uno).
NARRATION_SAMPLE_RATE = 48000

#: Tope de la transcripción que entra en el informe. La de un partido largo puede
#: ser enorme y, con ella entera, el prompt de la crónica no cabía en ``num_ctx``
#: (el modelo devolvía vacío y salía «se agotó el espacio de la respuesta»).
MAX_TRANSCRIPT_CHARS = 4000
#: Presupuesto del bloque de anotaciones del informe. Si hay más fotogramas que
#: líneas caben, se muestrean (primera, última y reparto) para que el informe cubra
#: TODO el vídeo: antes se recortaba por el final y la crónica solo narraba los
#: primeros minutos (todas las frases «al principio»).
MAX_NOTES_CHARS = 7000
#: Tope duro del informe completo que se le pasa al modelo (red de seguridad).
MAX_REPORT_CHARS = 14000

#: Estados/dicción que delatan un momento de JUEGO en una anotación del informe.
_ESTADOS_JUEGO_RE = re.compile(
    r"\b(jugando|saque|gol(?:es)?|parada|7\s*metros|penalti|lanzamiento|falta|"
    r"exclusi|ataque|contraataque|defiende|defensa|bloqueo|robo|recupera)\b",
    re.IGNORECASE,
)
#: Estados que delatan NO-juego (previo, descanso, final…): no son jugadas.
_ESTADOS_NO_JUEGO_RE = re.compile(
    r"\b(previo|saludo|calentamiento|descanso|entretiempo|final|tiempo\s*muerto)\b",
    re.IGNORECASE,
)


def _es_juego(texto: str) -> bool:
    """¿La anotación describe un momento de juego (y no un previo/descanso/final)?

    Se decide por el estado que aparece **primero** en la línea, porque el estado va
    al principio («ESTADO: JUGANDO…») mientras que las «Etiquetas» traen de todo
    («gol», «saludo»…): si se mirara la línea entera, casi todas las de juego se
    descartarían por el «saludo» de las etiquetas.
    """
    texto = texto or ""
    juego = _ESTADOS_JUEGO_RE.search(texto)
    no_juego = _ESTADOS_NO_JUEGO_RE.search(texto)
    if juego is None:
        return False
    if no_juego is None:
        return True
    return juego.start() < no_juego.start()


def _recortar(texto: str, tope: int) -> str:
    """Recorta ``texto`` a ``tope`` caracteres por palabra y avisa del corte."""
    texto = str(texto or "")
    if len(texto) <= tope:
        return texto
    corte = texto[:tope]
    if " " in corte:
        corte = corte[: corte.rfind(" ")]
    return corte.rstrip() + " […]"


def clamp_transcript(transcript: str, *, limit: int = MAX_TRANSCRIPT_CHARS) -> str:
    """Acota la transcripción que se incluye en el informe (avisa si recorta)."""
    return _recortar(transcript, limit)


def clamp_report(report: str, *, limit: int = MAX_REPORT_CHARS) -> str:
    """Acota el informe que recibe el modelo, para que quepa en la ventana."""
    return _recortar(report, limit)

#: Prompt por fotograma para la visión local. Pide **acción**, no estética: de lo
#: que se trata es de saber qué está pasando para poder contarlo después.
VISION_FRAME_PROMPT = (
    "Estás viendo un fotograma de un vídeo deportivo. En UNA frase corta, en "
    "español, empieza por el ESTADO del momento (PREVIO/CALENTAMIENTO/JUGANDO/"
    "SAQUE/GOL/PARADA/7 METROS/TIEMPO MUERTO/DESCANSO/FINAL) y luego di qué se ve: "
    "qué jugada o acción ocurre, quién parece protagonizarla (si se distingue) y el "
    "marcador si aparece en pantalla. Di QUÉ EQUIPO TIENE EL BALÓN (ese es el que "
    "ataca; el otro defiende). Si es un lanzamiento, di si acaba en GOL o en PARADA "
    "del portero. No describas el color ni la calidad de la imagen."
)

#: Título del bloque con los datos que sólo conoce el usuario (equipos, colores,
#: alineación, marcador…). Va en el prompt como *verdad*: la visión local ve
#: manchas de color, no nombres; esto es lo que convierte «un jugador» en «Álex».
CONTEXT_LABEL = "DATOS DEL PARTIDO (los aporta el usuario y son correctos)"
#: Ejemplo que se muestra en la caja de la pestaña, para que se vea de un vistazo
#: qué se puede escribir y con qué formato (una línea por dato).
CONTEXT_EXAMPLE = (
    "Fecha: 9/10/2026 a las 20:30\n"
    "Lugar: Polideportivo de Muskiz\n"
    "Equipos: PAN MENESA MUSKIZ camiseta azul (local) vs BAIGENE CORAZONISTAS "
    "camisetas rojas (visitante).\n"
    "Alineación PAN MENESA MUSKIZ: 2 ARANE PEREZ HERNANDEZ, 5 IRIA MORAGA GARCIA, "
    "6 JUGATX FERNANDEZ GONZALEZ...\n"
    "Alineación BAIGENE CORAZONISTAS: 3 AMANDA OTERO BENITO, 8 PAULA GARCIA "
    "VALENCIA, 9 AITANA ESTEBAN GARCIA...\n"
    "Marcador final: 42-29\n"
    "Nota: faltas del segundo tiempo del minuto 30 al 40, por error de cámara."
)


def context_block(context: str) -> str:
    """Normaliza los datos del partido: una línea por dato, sin huecos ni sobras.

    El usuario pega texto de cualquier manera (líneas en blanco, espacios de más,
    tabulaciones). Al modelo le llega ya limpio: mismo texto, sin ruido.
    """
    lineas = [" ".join(linea.split()) for linea in str(context or "").splitlines()]
    return "\n".join(linea for linea in lineas if linea)


#: Líneas de «Datos del partido» cuando van incrustadas en un informe (el Cuaderno
#: las incluye como bloque: «Equipos:», «Alineación …», «Marcador final:»…).
#: Dos puntos al final opcionales: hay notas del usuario sin «:» («Nota faltas del
#: segundo tiempo…») que también son datos del partido y no deben perderse.
_DATOS_RE = re.compile(
    r"^\s*(?:equipos?|alineaci[oó]n|jugadores?|jugadoras?|plantilla|convocatoria"
    r"|marcador|resultado|tanteo|notas?|deporte|disciplina"
    r"|lugar|pabell[oó]n|pista|cancha|sede|ciudad|fecha|hora)\b[^:]{0,40}:?",
    re.IGNORECASE,
)

#: Líneas de «Datos del partido» de las que sale la apertura y el marcador con
#: nombres de equipo. Se admite texto entre la etiqueta y el valor, y los dos puntos
#: son OPCIONALES: la gente escribe tanto «Lugar: …» como «Fecha 09/10/2026» (y sin
#: los dos puntos la fecha no salía en la apertura).
_LUGAR_RE = re.compile(
    r"^\s*(?:lugar|pabell[oó]n|pista|cancha|sede|ciudad)\b[^:]{0,40}?(?:[:\-]\s*|\s+)(.+)$",
    re.IGNORECASE,
)
_FECHA_RE = re.compile(
    r"^\s*(?:fecha|d[íi]a|hora)\b[^:]{0,40}?(?:[:\-]\s*|\s+)(.+)$", re.IGNORECASE
)
_EQUIPOS_RE = re.compile(
    r"^\s*equipos?\b[^:]{0,40}?(?:[:\-]\s*|\s+)(.+)$", re.IGNORECASE
)


def _campo(context: str, patron: re.Pattern) -> str:
    """Valor de la primera línea que casa con ``patron`` (sin la etiqueta)."""
    for linea in str(context or "").splitlines():
        match = patron.match(" ".join(linea.split()))
        if match:
            return match.group(1).strip(" .;")
    return ""


def teams_order(context: str) -> tuple[str, str]:
    """``(local, visitante)`` de la línea «Equipos: A (local) vs B (visitante)».

    Es lo que permite situar el marcador de pantalla (izquierda = local, derecha =
    visitante) y que la crónica no diga que va por delante quien no lo está. Si la
    línea no marca local/visitante, se toma el orden de escritura.
    """
    linea = _campo(context, _EQUIPOS_RE)
    if not linea:
        return "", ""
    trozos = re.split(
        r"\s+vs\.?\s+|\s+contra\s+|\s+-\s+|\s+–\s+", linea, flags=re.IGNORECASE
    )
    if len(trozos) < 2:
        return "", ""
    local = visitante = ""
    for indice, trozo in enumerate(trozos[:2]):
        limpio = re.sub(r"\([^)]*\)", " ", trozo).strip(" .,-–")
        bajo = trozo.lower()
        if "visitante" in bajo:
            visitante = limpio
        elif "local" in bajo:
            local = limpio
        elif indice == 0:
            local = local or limpio
        else:
            visitante = visitante or limpio
    return local, visitante


def datos_del_informe(report: str) -> str:
    """Datos del partido (equipos, alineación…) que van DENTRO de un informe.

    El Cuaderno escribe su contexto como bloque «Datos del partido» dentro del
    informe. Si el usuario no lo ha vuelto a escribir en la Crónica, aquí se
    recupera: con la alineación, la crónica llama a las jugadoras por su nombre de
    pila aunque la caja «Datos del partido» esté vacía.
    """
    return "\n".join(
        linea.strip()
        for linea in str(report or "").splitlines()
        if _DATOS_RE.match(linea)
    )


#: Línea del contexto que declara un marcador/tanteo. NO se pasa por fotograma: el
#: «Marcador final» se colaba en casi todas las descripciones, incluso al inicio.
#: Los equipos y colores sí se pasan (sitúan «el jugador azul»); el marcador real
#: lo lee el OCR.
_MARCADOR_LINEA_RE = re.compile(r"\bmarcador\b|\b\d{1,3}\s*[-:–]\s*\d{1,3}\b", re.IGNORECASE)


def _contexto_sin_marcador(context: str) -> str:
    """Contexto sin las líneas que declaran marcador/tanteo (una por dato)."""
    return "\n".join(
        linea
        for linea in context_block(context).splitlines()
        if not _MARCADOR_LINEA_RE.search(linea)
    )


def vision_frame_prompt(
    context: str = "", sport: str = "", instructions: str = ""
) -> str:
    """Prompt por fotograma; añade la disciplina, los equipos y las instrucciones.

    La **disciplina** aporta sus reglas y vocabulario (sin ellas la visión confunde
    deportes). Los equipos/colores convierten «un jugador» en «el jugador azul». Los
    nombres exactos y el marcador los pone la crónica, no la visión, y el «Marcador
    final» se omite a propósito (contaminaba todas las descripciones).

    ``instructions`` es lo que el usuario pide («qué quiero que cuentes»): así la
    visión no deja pasar momentos que la crónica necesita (p. ej. las PARADAS del
    portero) por describir solo «jugadores y balón».
    """
    partes = [VISION_FRAME_PROMPT]
    primer = nts.sport_primer(sport)
    if primer:
        partes.append(primer)
    equipos = rp.team_color_summary(context)
    if equipos:
        # Mapa color → equipo (LOCAL/VISITANTE): sin él la visión dice «el equipo
        # rojo» y confunde quién ataca y quién defiende.
        partes.append(
            "EQUIPOS SEGÚN EL COLOR DE CAMISETA: "
            f"{equipos}. El equipo que TIENE EL BALÓN es el que ataca; el otro "
            "defiende. Nómbralos por su equipo, no por el color."
        )
    limpio = _contexto_sin_marcador(context)
    if limpio:
        partes.append("Contexto del partido, para situar equipos y colores:\n" + limpio)
    # Las instrucciones del usuario describen la CRÓNICA (qué contar y en qué
    # orden), no cada fotograma: si se pasan por fotograma, el modelo intenta
    # «presentar la alineación» en TODOS y alucina listas enormes. Se aplican al
    # guion, no aquí.
    pedido = context_block(instructions)
    if pedido:
        partes.append(
            "Lo que el usuario quiere que destaques en cada fotograma (tenlo en "
            f"cuenta al describir):\n{pedido}"
        )
    return "\n\n".join(partes)


def frame_notes_text(
    notes: Sequence[tuple[float, str]], *, max_chars: int | None = None
) -> str:
    """Informe de los fotogramas: una línea por instante, con su minuto.

    Es lo que se le pasa al chat como «lo que se ve»: con el tiempo delante, el
    modelo puede escribir la crónica **con marcas de tiempo** en vez de inventarse
    el orden de las jugadas.

    Con ``max_chars`` el bloque se acota **de verdad**: se **muestrea** (primera,
    última y reparto uniforme) para cubrir TODO el vídeo y se recorta cada línea a
    lo que cabe, priorizando las anotaciones de juego. Antes solo se limitaba el
    número de líneas (``max_chars // 100``) dando por hecho líneas de ~100
    caracteres; con descripciones largas el informe se pasaba del tope,
    :func:`clamp_report` lo cortaba **por el final** y la crónica solo veía el
    primer tercio del partido — por eso salían 3 o 4 frases donde debía haber
    decenas.
    """
    limpias = [
        (float(t_s), " ".join(str(texto or "").split()))
        for t_s, texto in notes
        if str(texto or "").strip()
    ]
    if not limpias:
        return ""
    if not max_chars:
        return "\n".join(f"[{format_timestamp(t_s)}] {texto}" for t_s, texto in limpias)

    tope = max(1000, int(max_chars))
    max_lineas = max(10, tope // 100)
    if len(limpias) > max_lineas:
        # Se PRIORIZA el JUEGO: si el muestreo fuera a ciegas, se colarían
        # previos/saludos y el modelo no vería jugadas que narrar. Se conservan
        # primero las de juego (repartidas por todo el vídeo) y se rellena con el
        # resto, todo ordenado por tiempo.
        juego = [par for par in limpias if _es_juego(par[1])]
        if len(juego) >= max_lineas:
            limpias = nts.spread(juego, max_lineas)
        else:
            otras = [par for par in limpias if not _es_juego(par[1])]
            limpias = sorted(
                juego + nts.spread(otras, max_lineas - len(juego)),
                key=lambda par: par[0],
            )
    por_linea = max(60, tope // len(limpias) - 14)
    return "\n".join(
        f"[{format_timestamp(t_s)}] {texto[:por_linea]}" for t_s, texto in limpias
    )


def momentos_de_juego(report: str) -> int:
    """Nº de anotaciones ``[MM:SS]`` del informe que describen JUEGO.

    Sirve para pedir a la crónica un objetivo de frases acorde con lo que el
    informe realmente vio (no un tope abstracto): si hay 40 momentos de juego, que
    no escriba 3.
    """
    total = 0
    for linea in str(report or "").splitlines():
        if _RE_ANOTACION.search(linea) and _es_juego(linea):
            total += 1
    return total


def informe_report(data: dict) -> str:
    """Informe de la crónica a partir de un informe del **Cuaderno**.

    Usa las descripciones por fotograma que ya hizo el Cuaderno (y su
    transcripción, si la incluyó), con el mismo formato que el análisis propio de
    la crónica pero **sin volver a mirar el vídeo**. Los datos del partido que
    llevara el informe se añaden como contexto.
    """
    contexto = data.get("contexto", "")
    # Las anotaciones pueden venir de la visión, que sólo ve dorsales y colores:
    # con la alineación, el dorsal se traduce a nombre de pila antes de entrar en
    # el informe (así la crónica lee nombres, no «dorsal 7»).
    notas = [
        (t, rp.resolve_roster_names(texto, contexto))
        for t, texto in nts.informe_frames(data)
    ]
    partes: list[str] = []
    contexto = nts.context_block(contexto)
    if contexto:
        partes += [f"{nts.CONTEXT_LABEL}:\n{contexto}", ""]
    marcador = marcador_del_informe(data, context=str(data.get("contexto") or ""))
    if marcador:
        partes += [marcador, ""]
    partes += [
        "LO QUE SE VE EN EL VÍDEO (una línea por instante analizado):",
        frame_notes_text(notas, max_chars=MAX_NOTES_CHARS)
        or "(no se pudo describir ningún fotograma)",
    ]
    transcripcion = clamp_transcript(str(data.get("transcripcion", "")).strip())
    if transcripcion:
        partes += ["", "AUDIO DEL VÍDEO (transcripción literal):", transcripcion]
    return "\n".join(partes)


def perception_report(data: dict) -> str:
    """Informe de la crónica a partir de la percepción de **Comprensión**.

    Comprensión deja ``comprension.json`` con una anotación por fotograma (objetos
    de YOLO, etiquetas de CLIP, texto en pantalla por OCR), la transcripción y un
    resumen. La Crónica lo reutiliza **sin volver a mirar el vídeo**: es la vía
    más barata y rápida (no gasta visión) y, como Comprensión y Cuaderno comparten
    la rejilla de minutos, cada frase cae en el instante donde pasó el hecho.

    Es la ayuda que se espera de la pestaña Comprensión: analiza una vez y la
    Crónica lo aprovecha. Si además hay informe del Cuaderno, éste es preferible
    (trae las descripciones de la visión y la alineación).
    """
    contexto_crudo = str(data.get("contexto") or "").strip()
    notas = [
        (t, rp.resolve_roster_names(texto, contexto_crudo))
        for t, texto in understanding_frames(data)
    ]
    partes: list[str] = []
    contexto = nts.context_block(contexto_crudo)
    if contexto:
        partes += [f"{nts.CONTEXT_LABEL}:\n{contexto}", ""]
    resumen = str(data.get("resumen") or "").strip()
    if resumen:
        partes += ["RESUMEN DE COMPRENSIÓN (lo que ya se dedujo del vídeo):", resumen, ""]
    partes += [
        "LO QUE SE VE EN EL VÍDEO (una línea por instante analizado):",
        frame_notes_text(notas, max_chars=MAX_NOTES_CHARS)
        or "(no se pudo describir ningún fotograma)",
    ]
    transcripcion = clamp_transcript(str(data.get("transcripcion") or "").strip())
    if transcripcion:
        partes += ["", "AUDIO DEL VÍDEO (transcripción literal):", transcripcion]
    return "\n".join(partes)


def marcador_del_informe(data: dict, *, context: str = "") -> str:
    """Bloque de marcador/goles que dejó el Cuaderno (o cadena vacía).

    El informe puede traer ``marcador`` (serie de tanteos leídos por OCR) y
    ``goles``. Se reofrecen a la crónica como **datos reales**, para que no tenga
    que adivinar el resultado. Con ``context`` (los «Datos del partido») cada tanteo
    se etiqueta con el **nombre del equipo** que va a cada lado del marcador, y se
    avisa de que no puede decir que va por delante quien no lo esté.
    """
    local, visitante = teams_order(context or str(data.get("contexto") or ""))
    etiqueta = f" ({local} - {visitante})" if (local or visitante) else ""
    lineas: list[str] = []
    for punto in data.get("marcador") or []:
        if not isinstance(punto, dict):
            continue
        try:
            t_s = float(punto.get("t_s") or 0.0)
            izquierda = int(punto.get("izquierda") or 0)
            derecha = int(punto.get("derecha") or 0)
        except (TypeError, ValueError):
            continue
        lineas.append(f"[{format_timestamp(t_s)}] {izquierda}-{derecha}{etiqueta}")
    if not lineas:
        return ""
    cabecera = (
        "MARCADOR REAL LEÍDO POR OCR (izquierda-derecha según el marcador de "
        "pantalla; estos son los únicos goles que constan)"
    )
    if local and visitante:
        cabecera += (
            f": a la IZQUIERDA va {local} (LOCAL) y a la DERECHA {visitante} "
            "(VISITANTE)"
        )
    partes = [
        cabecera
        + ". Copia el resultado de aquí y NO digas que va por delante un equipo que "
        "no lo esté según esta serie:",
        "; ".join(lineas),
    ]
    goles = data.get("goles") or []
    if goles:
        partes.append(f"Goles detectados: {len(goles)}.")
    return "\n".join(partes)


@dataclass(frozen=True)
class ChronicleLine:
    """Una frase de la crónica y el segundo en el que se dice."""

    t_s: float
    texto: str

    def duration_s(self) -> float:
        """Duración estimada del subtítulo, por longitud del texto."""
        leido = len(self.texto) / CHARS_PER_SECOND
        return min(MAX_LINE_S, max(MIN_LINE_S, leido))

    def end_s(self, next_line: "ChronicleLine | None", total_s: float | None = None) -> float:
        """Final del subtítulo: su duración o el arranque de la siguiente frase."""
        fin = self.t_s + self.duration_s()
        if next_line is not None:
            fin = min(fin, max(self.t_s + MIN_LINE_S, next_line.t_s - MIN_GAP_S))
        if total_s:
            fin = min(fin, float(total_s))
        return max(self.t_s + 0.2, fin)


def clamp_lines(
    lines: Sequence[ChronicleLine],
    *,
    duration: float | None = None,
    max_lines: int = DEFAULT_MAX_LINES,
) -> tuple[ChronicleLine, ...]:
    """Ordena, recorta el texto, descarta lo que no cabe y acota el número.

    Un modelo local devuelve de todo: tiempos repetidos, frases fuera del vídeo o
    veinte líneas donde se pidieron ocho. Aquí se deja lo utilizable sin fallar.
    """
    limpias: list[ChronicleLine] = []
    for linea in lines:
        texto = " ".join((linea.texto or "").split())[:MAX_LINE_CHARS].strip()
        if len(texto) < MIN_LINE_CHARS:
            continue
        if duration is not None and linea.t_s >= float(duration):
            continue
        limpias.append(ChronicleLine(max(0.0, float(linea.t_s)), texto))
    limpias.sort(key=lambda linea: linea.t_s)
    tope = max(1, min(int(max_lines or DEFAULT_MAX_LINES), MAX_LINES))
    return tuple(limpias[:tope])


def chronicle_srt(
    lines: Sequence[ChronicleLine], *, duration: float | None = None
) -> str:
    """SRT de la crónica (se quema con el estilo que elija el usuario)."""
    tramos = list(lines)
    return build_srt(
        [
            (linea.t_s, linea.end_s(tramos[i + 1] if i + 1 < len(tramos) else None, duration), linea.texto)
            for i, linea in enumerate(tramos)
        ]
    )


def chronicle_markdown(
    lines: Sequence[ChronicleLine],
    *,
    video_name: str = "",
    duration: float | None = None,
) -> str:
    """Entrega legible de la crónica (el texto con sus marcas de tiempo)."""
    titulo = f"# Crónica · {video_name}" if video_name else "# Crónica"
    cabecera = [titulo, ""]
    if duration:
        cabecera += [f"Duración: {format_timestamp(duration)}", ""]
    if not lines:
        return "\n".join(cabecera + ["_Sin frases._", ""])
    cuerpo = [
        f"- **{format_timestamp(linea.t_s)}** — {linea.texto}" for linea in lines
    ]
    return "\n".join(cabecera + cuerpo + [""])


def chronicle_prompt(
    report: str,
    *,
    language: str = DEFAULT_LANGUAGE,
    max_lines: int = DEFAULT_MAX_LINES,
    duration_s: float | None = None,
    video_name: str = "",
    context: str = "",
    sport: str = "",
    instructions: str = "",
) -> str:
    """Prompt con el que el chat escribe la crónica a partir del informe.

    ``context`` son los datos que sólo conoce el usuario (equipos, colores,
    alineación, marcador). Con ellos la crónica llama a los jugadores por su
    nombre en vez de hablar de «un jugador»; sin ellos, se escribe igual.
    ``sport`` añade las **reglas y el vocabulario** de la disciplina, para que la
    crónica interprete la jugada (golpe franco, exclusión, 7 metros…) y no la
    describa con vocabulario de otro deporte.
    ``instructions`` es lo que el usuario pide contar y en qué orden (p. ej.
    «presenta la alineación al principio y luego céntrate en goles, paradas y
    7 metros»); manda por encima del reparto uniforme de frases.
    """
    dur = f"Duración del vídeo: {format_timestamp(duration_s)}\n" if duration_s else ""
    nombre = f"Vídeo: {video_name}\n" if video_name else ""
    idioma = (language or DEFAULT_LANGUAGE).strip()
    tope = max(1, min(int(max_lines or DEFAULT_MAX_LINES), MAX_LINES))
    # Momentos de juego que el informe realmente vio: con ellos se le pide a la
    # crónica un objetivo concreto (no un tope abstracto que el modelo ignoraba).
    momentos = momentos_de_juego(report)
    limpio = context_block(context)
    datos = ""
    if limpio:
        datos = (
            f"\n{CONTEXT_LABEL}:\n{limpio}\n"
            "Esos datos son correctos: usa EXACTAMENTE esos nombres, dorsales, "
            "colores y marcador, sin cambiarlos ni inventar otros. Llama a cada "
            "jugadora por su NOMBRE DE PILA, no por su dorsal ni por el color de la "
            "camiseta (el dorsal y el color sólo sirven para identificarla). Si en "
            "el informe no se distingue a nadie, no le pongas nombre.\n"
        )
    # Si la caja va vacía, la alineación puede venir dentro del informe (el Cuaderno
    # la incluye como «Datos del partido»): se usa para traducir dorsal → nombre.
    roster = rp.roster_block(limpio or datos_del_informe(report))
    bloque_roster = f"\n{roster}\n" if roster else ""
    reglas = nts.sport_primer(sport)
    bloque_reglas = ""
    if reglas:
        bloque_reglas = (
            "\nREGLAS Y VOCABULARIO DE LA DISCIPLINA (para interpretar lo que se ve; "
            "no las expliques, solo úsalas para describir bien la jugada):\n"
            f"{reglas}\n"
        )
    pedido = context_block(instructions)
    bloque_instrucciones = ""
    if pedido:
        bloque_instrucciones = (
            f"\n{sp.INSTRUCTIONS_LABEL}:\n{pedido}\n"
            "Estas instrucciones MANDA: ordénate como pide el usuario, aunque no "
            "reparta las frases de forma uniforme.\n"
        )
    local, visitante = teams_order(limpio or datos_del_informe(report))
    bloque_equipos = ""
    if local or visitante:
        bloque_equipos = "\nEQUIPOS Y MARCADOR (no los cambies):\n"
        if local:
            bloque_equipos += f"- LOCAL (izquierda del marcador): {local}\n"
        if visitante:
            bloque_equipos += f"- VISITANTE (derecha del marcador): {visitante}\n"
        bloque_equipos += (
            "El equipo que TIENE EL BALÓN ataca; el otro defiende. No digas que "
            "ataca quien no tiene el balón ni que defiende quien lo lleva.\n"
            "Copia el resultado del MARCADOR REAL del informe y NO digas que va por "
            "delante un equipo que no lo esté en esa serie: si el marcador no "
            "confirma el liderato de nadie, no hables del resultado.\n"
        )
    return (
        "Eres comentarista deportivo. Escribe la CRÓNICA del vídeo a partir del "
        f"informe de abajo, en {idioma}, con frases cortas y ritmo de retransmisión.\n"
        f"{nombre}{dur}{datos}{bloque_roster}{bloque_reglas}{bloque_equipos}"
        f"{bloque_instrucciones}"
        f"Escribe como máximo {tope} frases. Cada frase va ANCLADA al momento exacto "
        "del informe en el que ocurre el hecho que cuenta (no a un tiempo inventado).\n"
        f"Apunta a cubrir el partido: el informe trae ~{momentos} momentos de JUEGO; "
        "escribe una frase por cada uno que aporte algo distinto (no solo goles y "
        "marcador: también saques, ataques, defensas y paradas), sin pasar del "
        f"máximo de {tope} frases. Es un error habitual quedarse en 3 o 4 frases "
        "cuando hay decenas de momentos anotados.\n"
        "Por eso CADA frase lleva OBLIGATORIAMENTE:\n"
        "- «ref»: el [MM:SS] EXACTO de la anotación del informe en la que te basas "
        "(cópialo tal cual, con corchetes o sin ellos).\n"
        "- «t_s»: ese mismo instante pasado a segundos.\n"
        "- «texto»: la frase.\n"
        "No inventes tiempos ni repartas las frases por tu cuenta. Si una idea no "
        "está en ninguna anotación [MM:SS] del informe, NO escribas esa frase.\n"
        "IDENTIDAD: cuando el informe ya haya identificado a una jugadora (por su "
        "dorsal y su nombre de la alineación), sigue llamándola por su nombre en las "
        "frases siguientes aunque en esa anotación no se le lea el dorsal: es la "
        "misma jugadora. Si no lo tienes claro, di «una jugadora del equipo azul» o "
        "«del rojo»; NUNCA inventes un nombre.\n"
        "Cubre TODO el vídeo: reparte las frases por el partido (principio, medio y "
        "final), no las amontones al principio.\n"
        "PRIORIDAD: las jugadas decisivas primero (GOLES, PARADAS del portero y "
        "penaltis / 7 metros), por delante de cualquier otra. OBLIGATORIO: si el "
        "informe trae anotaciones con GOL, PARADA o 7 METROS, dedica UNA frase a "
        "CADA una (con su ref) antes que a cualquier otro momento.\n"
        "VERIFICA al final: tu crónica debe tener UNA frase por CADA gol del "
        "MARCADOR REAL (ni agrupados ni saltados) y una por cada PARADA y 7 METROS "
        "anotados. Si el marcador trae N goles, escribe N frases de gol.\n"
        "No abras con la sede, la fecha ni la alineación: la app ya añade esas "
        "frases al principio. Empieza directamente por la acción.\n"
        "Cuenta solo momentos de JUEGO (no descansos, saludos, tiempos muertos ni "
        "celebraciones largas).\n"
        "Devuelve SÓLO un JSON con esta forma exacta:\n"
        '{"cronica": [{"ref": "00:00", "t_s": 0, "texto": "Arranca el partido"}, '
        '{"ref": "00:12", "t_s": 12, "texto": "¡Gol de Álex!"}]}\n'
        "No inventes jugadas que no aparezcan en el informe: si algo no se ve, no "
        "se cuenta. No repitas la misma idea en dos frases.\n\n"
        "INFORME DEL VÍDEO:\n"
        f"{clamp_report(report)}"
    )


def parse_chronicle(
    raw: str,
    *,
    duration: float | None = None,
    max_lines: int = DEFAULT_MAX_LINES,
    context: str = "",
) -> tuple[ChronicleLine, ...]:
    """Convierte la respuesta del chat en frases con tiempo.

    Acepta el JSON pedido (:data:`chronicle_prompt`) y, si el modelo contesta en
    texto, líneas «minuto:segundo frase» (el mismo formato que los rótulos). Con
    ``context`` (y su alineación), cada frase que mencione un dorsal pasa a usar el
    **nombre de pila** de esa jugadora. Lanza ``ValueError`` si no hay nada
    aprovechable, para que la UI lo diga en vez de montar un vídeo sin voz.
    """
    lineas: list[ChronicleLine] = []
    datos = None
    try:
        datos = nts.extract_json(raw)
    except nts.NotesError:
        datos = None

    if isinstance(datos, dict):
        entradas = datos.get("cronica") or datos.get("lineas") or datos.get("frases")
        if isinstance(entradas, list):
            for entrada in entradas:
                if not isinstance(entrada, dict):
                    continue
                texto = str(entrada.get("texto") or entrada.get("frase") or "").strip()
                # El anclaje real es «ref» (el [MM:SS] del informe); «t_s» es sólo
                # el respaldo si el modelo omite la referencia.
                referencia = entrada.get(
                    "ref", entrada.get("referencia", entrada.get("momento"))
                )
                t_s = parse_timestamp(referencia) if referencia is not None else None
                if t_s is None:
                    tiempo = entrada.get(
                        "t_s", entrada.get("segundo", entrada.get("time"))
                    )
                    try:
                        t_s = float(tiempo)
                    except (TypeError, ValueError):
                        continue
                lineas.append(ChronicleLine(t_s, texto))

    if not lineas:
        # Respuesta en texto: «0:12 Gol de Álex» (lo mismo que los rótulos).
        try:
            lineas = [
                ChronicleLine(rotulo.t_s, rotulo.texto)
                for rotulo in vep.parse_rolulos(raw)
            ]
        except ValueError as exc:
            raise ValueError(
                "El motor no devolvió una crónica con tiempos. Vuelve a intentarlo "
                "o usa un modelo de chat más grande."
            ) from exc

    if context:
        lineas = [
            ChronicleLine(linea.t_s, rp.resolve_roster_names(linea.texto, context))
            for linea in lineas
        ]
    limpias = clamp_lines(lineas, duration=duration, max_lines=max_lines)
    if not limpias:
        raise ValueError(
            "La crónica quedó vacía (frases demasiado cortas o fuera del vídeo)."
        )
    return limpias


_RE_ANOTACION = re.compile(r"\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]")


def report_timestamps(report: str) -> list[float]:
    """Instantes (segundos) que aparecen como ``[MM:SS]`` (o ``[H:MM:SS]``)."""
    instantes: list[float] = []
    for m in _RE_ANOTACION.finditer(report or ""):
        primero, segundo, tercero = int(m.group(1)), int(m.group(2)), m.group(3)
        if tercero is not None:
            instantes.append(primero * 3600 + segundo * 60 + int(tercero))
        else:
            instantes.append(primero * 60 + segundo)
    return instantes


#: Palabras que delatan que una frase narra un gol (verificación de cobertura).
_GOAL_WORDS_RE = re.compile(
    r"\b(gol|marca|anota|empata|adelanta|ampl[íi]a|recorta|acorta|suma|"
    r"remata|convierte|transforma|hace el)\b",
    re.IGNORECASE,
)


def _detected_goals(report: str) -> int | None:
    """Nº de goles que declara el informe (del bloque de marcador), o ``None``."""
    match = re.search(
        r"(?:total de\s+)?goles detectados:\s*(\d+)",
        str(report or ""),
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1))
    return None


def coverage_warnings(report: str, lines: Sequence[ChronicleLine]) -> list[str]:
    """Avisos si la crónica se dejó puntos especiales sin narrar.

    Compara los goles que el informe declara (del marcador OCR) con las frases de
    la crónica: si hay menos frases de gol que goles detectados, avisa para que el
    usuario sepa que algo se quedó fuera (mejor un aviso que un vídeo cojo).
    """
    avisos: list[str] = []
    total = _detected_goals(report)
    narrados = sum(1 for linea in lines if _GOAL_WORDS_RE.search(linea.texto))
    if total is not None and narrados < total:
        avisos.append(
            f"La crónica narra {narrados} goles de los {total} que constan en el "
            "marcador OCR: faltan goles por contar."
        )
    return avisos


def snap_lines(
    lines: Sequence[ChronicleLine],
    instants: Sequence[float],
    *,
    tolerance: float = 45.0,
) -> tuple[ChronicleLine, ...]:
    """Alinea cada frase al instante anotado más cercano (evita «hablar a destiempo»).

    El modelo asigna los tiempos con deriva; aquí se enganchan al momento que sí se
    vio (los ``[MM:SS]`` del informe) si está dentro de ``tolerance``. Si dos frases
    caen en el mismo instante, se deja solo la primera (no se pisan).
    """
    if not instants or not lines:
        return tuple(lines)
    orden = sorted(float(t) for t in instants)
    ajustadas: list[ChronicleLine] = []
    for linea in lines:
        i = bisect.bisect_left(orden, linea.t_s)
        candidatos = [orden[j] for j in (i - 1, i) if 0 <= j < len(orden)]
        mejor = min(candidatos, key=lambda t: abs(t - linea.t_s)) if candidatos else None
        if mejor is not None and abs(mejor - linea.t_s) <= tolerance:
            ajustadas.append(ChronicleLine(float(mejor), linea.texto))
        else:
            ajustadas.append(linea)
    ajustadas.sort(key=lambda linea: linea.t_s)
    unicas: list[ChronicleLine] = []
    vistas: set[float] = set()
    for linea in ajustadas:
        clave = round(linea.t_s, 3)
        if clave in vistas:
            continue
        vistas.add(clave)
        unicas.append(linea)
    return tuple(unicas)


def anchor_lines(
    lines: Sequence[ChronicleLine],
    instants: Sequence[float],
) -> tuple[ChronicleLine, ...]:
    """Coloca cada frase en el instante anotado más cercano **que esté libre**.

    La crónica sólo puede narrar lo que Comprensión/Cuaderno anotaron: cada frase
    cae en la **misma rejilla** que el Cuaderno (mismos minutos) y desaparece el
    desfase de inventarse tiempos. Si dos frases apuntan al mismo instante, la
    segunda pasa al siguiente instante anotado libre: así **no se pierde ninguna
    frase** (antes se descartaban las repetidas y la crónica se quedaba en 4 líneas
    cuando el informe tenía muchos fotogramas con marcas repetidas).
    """
    if not instants or not lines:
        return tuple(lines)
    orden = sorted(float(t) for t in instants)
    ocupados: set[int] = set()
    ajustadas: list[ChronicleLine] = []
    for linea in sorted(lines, key=lambda x: x.t_s):
        i = bisect.bisect_left(orden, linea.t_s)
        elegido: int | None = None
        for salto in range(len(orden)):
            for j in (i + salto, i - salto):
                if 0 <= j < len(orden) and j not in ocupados:
                    elegido = j
                    break
            if elegido is not None:
                break
        if elegido is None:  # más frases que instantes: se deja como estaba
            ajustadas.append(linea)
            continue
        ocupados.add(elegido)
        ajustadas.append(ChronicleLine(orden[elegido], linea.texto))
    ajustadas.sort(key=lambda linea: linea.t_s)
    return tuple(ajustadas)


#: Ritmo conservador (caracteres/segundo) al estimar lo que dura una frase locutada:
#: quedarse corto hace que una frase pise a la siguiente (la voz de la alineación del
#: Muskiz se solapaba con la de los Corazonistas). El worker mide el audio real de la
#: apertura y la reajusta; esto es sólo el punto de partida.
_SPEECH_CHARS_PER_S = 11.0


def _duracion_hablada(texto: str) -> float:
    """Segundos estimados que se tarda en locutar ``texto`` (con margen)."""
    return max(MIN_LINE_S, len(texto) / _SPEECH_CHARS_PER_S) + 0.6


def opening_lines(context: str, *, start_s: float = 0.0) -> tuple[ChronicleLine, ...]:
    """Apertura de la crónica: sede, fecha y alineación de los **dos** equipos.

    Se compone **con reglas** a partir de «Datos del partido» (Lugar, Fecha,
    Equipos, Alineación…), no con el modelo: así la crónica siempre empieza
    presentando el polideportivo, la fecha y las alineaciones, y esas frases no
    dependen de que el modelo las copie ni de que exista una anotación [MM:SS] en
    ese momento. Si no hay ningún dato, devuelve ``()``.
    """
    textos: list[str] = []
    lugar = _campo(context, _LUGAR_RE)
    if lugar:
        textos.append(f"Estamos en {lugar}.")
    fecha = _campo(context, _FECHA_RE)
    if fecha:
        textos.append(f"Partido del {fecha}.")
    equipos = _campo(context, _EQUIPOS_RE)
    if equipos:
        textos.append(f"Se enfrentan {equipos}.")
    jugadores = rp.parse_roster(context)
    equipos_roster: list[str] = []
    for jugador in jugadores:
        if jugador.equipo not in equipos_roster:
            equipos_roster.append(jugador.equipo)
    for equipo in equipos_roster:
        plantilla = [f"{j.dorsal} {j.nombre}" for j in jugadores if j.equipo == equipo]
        if plantilla:
            textos.append(f"Alineación del {equipo}: " + ", ".join(plantilla) + ".")
    if not textos:
        return ()
    lineas: list[ChronicleLine] = []
    instante = float(start_s)
    for texto in textos:
        texto = _recortar(texto, MAX_LINE_CHARS)
        lineas.append(ChronicleLine(instante, texto))
        instante += _duracion_hablada(texto)
    return tuple(lineas)


def prepend_opening(
    opening: Sequence[ChronicleLine], lines: Sequence[ChronicleLine]
) -> tuple[ChronicleLine, ...]:
    """Pone la apertura delante y descarta las frases que se pisarían con ella.

    Las frases del modelo caen en instantes anotados; si alguna empieza antes de que
    termine la apertura, al locutar se solaparían (dos voces a la vez). Se quitan
    esas primeras frases y se conserva el resto; si eso dejara la crónica vacía, se
    devuelven todas (mejor una crónica con solape que una sin narración).
    """
    if not opening:
        return tuple(lines)
    apertura = tuple(opening)
    fin = apertura[-1].t_s + _duracion_hablada(apertura[-1].texto)
    resto = [linea for linea in lines if linea.t_s >= fin]
    if not resto:
        resto = list(lines)
    return apertura + tuple(resto)


def place_lines(
    lines: Sequence[ChronicleLine],
    instants: Sequence[float],
    *,
    tolerance: float = 3.0,
    min_gap: float = 0.5,
) -> tuple[ChronicleLine, ...]:
    """Coloca las frases **en su momento** (sin retrasos ni saltos).

    El modelo ya devuelve el ``ref`` del informe; aquí sólo se corrige la deriva si
    hay una anotación hasta ``tolerance`` segundos **antes** de la frase (así nunca
    se retrasa: no se oye la frase cuando el balón ya lo tiene el otro equipo). Si
    no, se respeta el instante del modelo: ``anchor_lines`` movía cada frase al
    instante libre más cercano y una del calentamiento podía acabar en el minuto
    3:40. Después se garantiza una separación mínima ``min_gap`` entre frases
    consecutivas, empujando sólo lo justo, para que no se pisen al locutar.
    """
    if not lines:
        return ()
    orden = sorted(float(t) for t in instants) if instants else []
    ajustadas: list[ChronicleLine] = []
    for linea in sorted(lines, key=lambda x: x.t_s):
        mejor = float(linea.t_s)
        if orden:
            # Sólo se engancha a una anotación que esté ANTES o en el mismo instante
            # (nunca después, hasta ``tolerance``): así la frase no se retrasa y no
            # se oye cuando la acción ya ha pasado.
            i = bisect.bisect_right(orden, linea.t_s)
            previas = [orden[j] for j in range(i)]
            if previas:
                anterior = previas[-1]
                if linea.t_s - anterior <= tolerance:
                    mejor = float(anterior)
        if ajustadas and mejor < ajustadas[-1].t_s + min_gap:
            mejor = ajustadas[-1].t_s + min_gap
        ajustadas.append(ChronicleLine(mejor, linea.texto))
    return tuple(ajustadas)


def narration_filter_complex(
    entries: Sequence[tuple[float, object]], *, sample_rate: int = NARRATION_SAMPLE_RATE
) -> str:
    """Grafo que coloca cada WAV en su segundo y los mezcla en una pista.

    Cada entrada es ``(t_s, ruta)`` y ocupa una entrada de ffmpeg, en el mismo
    orden (``[0:a]``, ``[1:a]``…). Se normaliza frecuencia y formato **antes** de
    ``amix`` (Piper y Kokoro no usan la misma) y ``normalize=0`` evita que ffmpeg
    baje el volumen al sumar pistas: la voz la tiene que oír el usuario.
    """
    if not entries:
        raise ValueError("La narración no tiene ninguna frase.")
    partes: list[str] = []
    etiquetas: list[str] = []
    for indice, (t_s, _ruta) in enumerate(entries):
        ms = max(0, int(round(float(t_s) * 1000)))
        partes.append(
            f"[{indice}:a]aresample={int(sample_rate)},"
            "aformat=sample_fmts=fltp:channel_layouts=mono,"
            f"adelay={ms}:all=1[voz{indice}]"
        )
        etiquetas.append(f"[voz{indice}]")
    entradas = "".join(etiquetas)
    partes.append(f"{entradas}amix=inputs={len(entries)}:normalize=0:duration=longest[a]")
    return ";".join(partes)


def narration_command(
    ffmpeg: str,
    entries: Sequence[tuple[float, object]],
    output: object,
    *,
    sample_rate: int = NARRATION_SAMPLE_RATE,
    timeout: float | None = None,
) -> list[str]:
    """Comando de ffmpeg que compone la pista de narración (una sola pasada)."""
    if not entries:
        raise ValueError("La narración no tiene ninguna frase.")
    argv = [ffmpeg, "-y", "-hide_banner"]
    for _t_s, ruta in entries:
        argv += ["-i", str(ruta)]
    argv += [
        "-filter_complex", narration_filter_complex(entries, sample_rate=sample_rate),
        "-map", "[a]",
        "-ar", str(int(sample_rate)),
        # Mezcla a WAV PCM: el Ensamblador lo vuelve a codificar, así que aquí no
        # interesa perder calidad con un códec con pérdidas intermedio.
        "-c:a", "pcm_s16le",
        str(output),
    ]
    return argv


def default_output_name(video: object) -> str:
    """Nombre base de los entregables de la crónica (``peli_cronica``)."""
    return f"{Path(str(video)).stem}_cronica"
