"""Retoque de vídeo: filtros de ffmpeg (color, denoise, nitidez, FPS, velocidad).

Construye cadenas de filtros de ffmpeg para corregir el color, limpiar ruido,
enfocar, interpolar fotogramas y **cambiar la velocidad** (cámara lenta/rápida,
con las repeticiones de un momento). Sin Qt ni subprocess: solo texto, para poder
testearlo y reutilizarlo. El escalado con IA se hace aparte (fotograma a
fotograma con Real-ESRGAN) y vive en el worker.

Sobre la velocidad hay dos detalles que valen la pena tener presentes:

- El audio **no** se puede copiar: si el vídeo va a la mitad de velocidad y el
  audio no, se desincronizan. Se reajusta con ``atempo`` (``audio_speed_filter``),
  que ffmpeg sólo admite entre 0.5 y 2.0 por paso, así que 0.25x se encadena.
- Una cámara lenta «de verdad» necesita fotogramas que no existen: con
  ``smooth=True`` se interpolan antes de estirar el tiempo
  (``fps_filter`` + ``setpts``), que es lo que convierte el efecto de diapositivas
  en movimiento continuo a cambio de mucho más tiempo de proceso.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from services.video_understanding import build_srt

# ---- Rangos de los controles --------------------------------------

BRIGHTNESS_MIN, BRIGHTNESS_MAX, BRIGHTNESS_DEFAULT = -0.5, 0.5, 0.0
CONTRAST_MIN, CONTRAST_MAX, CONTRAST_DEFAULT = 0.0, 3.0, 1.0
SATURATION_MIN, SATURATION_MAX, SATURATION_DEFAULT = 0.0, 3.0, 1.0
GAMMA_MIN, GAMMA_MAX, GAMMA_DEFAULT = 0.1, 3.0, 1.0
SHARPEN_MIN, SHARPEN_MAX, SHARPEN_DEFAULT = 0.0, 2.0, 0.0
#: Calidez: de −100 (frío, azulado) a +100 (cálido, anaranjado); 0 = neutro.
TEMPERATURE_MIN, TEMPERATURE_MAX, TEMPERATURE_DEFAULT = -100.0, 100.0, 0.0
#: Kelvin del filtro ``colortemperature``: 6500 es neutro; por debajo, cálido.
TEMPERATURE_NEUTRAL_K = 6500
TEMPERATURE_WARM_K = 3800
TEMPERATURE_COOL_K = 9500

#: Niveles de reduce-ruido (hqdn3d). ``off`` = no se aplica.
DENOISE_LEVELS: tuple[str, ...] = ("off", "light", "medium", "strong")
DENOISE_FILTERS: dict[str, str] = {
    "light": "hqdn3d=1.5:1.5:6:6",
    "medium": "hqdn3d=3:3:12:12",
    "strong": "hqdn3d=6:6:18:18",
}

#: Calidades del codificador H.264 (menor = mejor).
CRF_MIN, CRF_MAX, CRF_DEFAULT = 0, 35, 18

# ---- Velocidad (cámara lenta y rápida) ----------------------------

#: Rango del multiplicador de reproducción: <1 cámara lenta, >1 acelerado.
SPEED_MIN, SPEED_MAX, SPEED_DEFAULT = 0.25, 2.0, 1.0
#: Paso del control de velocidad.
SPEED_STEP = 0.05

#: Velocidad de la última repetición de un *replay* (la repetición lenta).
REPLAY_SLOW_SPEED = 0.5
#: Repeticiones admitidas por replay (1 = sólo recortar el momento).
REPLAY_MIN_TIMES, REPLAY_MAX_TIMES, REPLAY_TIMES_DEFAULT = 1, 6, 3
#: Duración por defecto del fragmento que se repite, en segundos.
REPLAY_CLIP_DEFAULT_S = 6.0

#: Tope de fps interpolados al suavizar una cámara lenta (más no se aprecia).
SMOOTH_MAX_FPS = 120
#: fps supuestos cuando la sonda no puede leer los del vídeo.
SMOOTH_FALLBACK_FPS = 30.0
#: Escalones que admite ``atempo``; fuera de ahí se encadenan varios.
ATEMPO_MIN, ATEMPO_MAX = 0.5, 2.0

#: Zoom para acercar la jugada (1.0 = sin tocar el encuadre).
ZOOM_MIN, ZOOM_MAX, ZOOM_DEFAULT = 1.0, 3.0, 1.0
#: Paso del control de zoom.
ZOOM_STEP = 0.1

# ---- Compresión (recodificar más pequeño) --------------------------

#: Calidades ofrecidas: (clave, etiqueta, CRF de CPU, CQ de GPU).
#:
#: El valor de GPU **no** puede ser el mismo que el de CPU: la escala ``-cq`` de
#: NVENC no equivale al ``-crf`` de x264. Medido con un trozo real de AVCHD
#: (1080p50, 25,4 Mbps de origen, 34 s):
#:
#:   x264 crf 20 -> 58,0 MB · 13,6 Mbps      nvenc cq 26 -> 59,9 MB · 14,1 Mbps
#:   x264 crf 23 -> 35,3 MB ·  8,3 Mbps      nvenc cq 30 -> 33,4 MB ·  7,8 Mbps
#:
#: Es decir, el CQ va unos **6 puntos por encima** del CRF para dejar un archivo
#: parecido. A igualdad de tamaño, NVENC da algo menos de calidad que x264 (es más
#: rápido, no mejor), así que para la entrega final conviene CPU si el tamaño es
#: holgado.
COMPRESSION_QUALITIES: tuple[tuple[str, str, int, int], ...] = (
    ("alta", "Alta (archivo grande)", 20, 26),
    ("media", "Equilibrada", 23, 29),
    ("pequena", "Pequeña (para compartir)", 26, 32),
    ("minima", "Mínima (móvil y mensajería)", 30, 36),
)
COMPRESSION_DEFAULT = "media"

#: Codificadores: CPU (libx264, calidad por bitrate) o GPU (NVENC, mucho más rápido).
ENCODER_CPU = "cpu"
ENCODER_GPU = "gpu"

#: Bitrate del audio al comprimir (el AC-3 de cámara se pasa a AAC para que lo
#: abra cualquier reproductor y para que el archivo pese lo que debe).
COMPRESSION_AUDIO_BITRATE = "192k"


def quality_value(key: str, encoder: str = ENCODER_CPU) -> int:
    """CRF (CPU) o CQ (GPU) de una calidad por su clave.

    Clave desconocida -> la calidad por defecto; así una preferencia vieja o mal
    escrita nunca rompe una compresión.
    """
    elegida = next(
        (fila for fila in COMPRESSION_QUALITIES if fila[0] == key),
        next(fila for fila in COMPRESSION_QUALITIES if fila[0] == COMPRESSION_DEFAULT),
    )
    return elegida[3] if encoder == ENCODER_GPU else elegida[2]


def has_nvenc(encoders_text: str) -> bool:
    """¿El ffmpeg instalado trae el codificador H.264 por GPU de NVIDIA?

    Se comprueba sobre la salida de ``ffmpeg -encoders`` (texto), para poder
    probarlo sin ejecutar nada.
    """
    return "h264_nvenc" in (encoders_text or "")


def batch_progress(index: int, inner_percent: float, total: int) -> int:
    """Porcentaje global de un lote (``inner_percent`` es el del fichero en curso).

    Se emite progreso **por fichero** dentro de un total: sin esto, comprimir diez
    trozos daría diez «100 %» y la barra volvería a cero diez veces.
    """
    total = max(1, int(total))
    dentro = min(100.0, max(0.0, float(inner_percent))) / 100.0
    return int(min(100.0, (int(index) + dentro) / total * 100))


#: Volumen del audio, en porcentaje: 100 = como está, 0 = silencio, 200 = el doble.
VOLUME_MIN, VOLUME_MAX, VOLUME_DEFAULT = 0, 200, 100


def volume_filter(percent) -> str | None:
    """``volume=0.600`` para dejar el audio al 60 %, o ``None`` si está al 100 %.

    Se usa para bajar (o subir) el sonido de un vídeo sin tocar nada más: es la
    forma rápida de arreglar una grabación que se oye demasiado alta.
    """
    try:
        valor = float(percent)
    except (TypeError, ValueError):
        return None
    if abs(valor - VOLUME_DEFAULT) < 1e-6:
        return None
    valor = max(VOLUME_MIN, min(VOLUME_MAX, valor))
    return f"volume={valor / 100.0:.3f}"


def parse_out_time(linea: str) -> float | None:
    """Segundos ya procesados a partir de una línea de ``ffmpeg -progress``.

    ffmpeg escribe ``out_time_us=…`` (en versiones viejas, ``out_time_ms``) con
    **microsegundos**, no milisegundos: de ahí el 10⁶. Devuelve ``None`` para
    cualquier otra línea, para poder ignorarla sin comprobar nada más.
    """
    clave, _, valor = str(linea or "").partition("=")
    if clave.strip() not in ("out_time_us", "out_time_ms"):
        return None
    try:
        return float(valor.strip()) / 1_000_000.0
    except ValueError:
        return None


def progress_percent(seconds, duration, *, stage=(0.0, 100.0)) -> float:
    """Porcentaje terminado, trasladado a un tramo de la barra (``stage``).

    ``seconds`` es lo ya procesado y ``duration`` el total; el resultado se acota a
    [0, 1] para que el último fotograma no dé un 103 %. Sin duración fiable no hay
    porcentaje y se devuelve el inicio del tramo.
    """
    inicio, fin = float(stage[0]), float(stage[1])
    total = float(duration or 0.0)
    if total <= 0.0:
        return inicio
    fraccion = max(0.0, min(1.0, float(seconds) / total))
    return inicio + (fin - inicio) * fraccion


def encoder_args(encoder: str = ENCODER_CPU, quality: int = 23) -> list[str]:
    """Argumentos del codificador de vídeo según CPU o GPU.

    En GPU se usa calidad constante (``-rc vbr`` con ``-cq`` y ``-b:v 0``), que es
    el equivalente a ``-crf`` en CPU. Ojo: a igual número, NVENC deja un archivo
    algo peor que libx264 (es más rápido, no mejor); si el tamaño importa y la
    calidad también, sube un punto el valor o usa CPU para la entrega final.
    """
    valor = str(int(quality))
    if encoder == ENCODER_GPU:
        return [
            "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq",
            "-rc", "vbr", "-cq", valor, "-b:v", "0",
            "-pix_fmt", "yuv420p",
        ]
    return [
        "-c:v", "libx264", "-crf", valor, "-preset", "medium", "-pix_fmt", "yuv420p",
    ]

# ---- Unir grabaciones troceadas ------------------------------------

#: Timeout de la unión: copia de streams, pero una grabación larga tarda en leerse.
JOIN_TIMEOUT = 3600
#: Bitrate del audio AAC al unir (las cámaras graban AC-3 a 256k).
JOIN_AUDIO_BITRATE = "192k"
#: Segunda vuelta de la unión (igualar trozos): recodifica, así que da margen.
JOIN_REENCODE_TIMEOUT = 14400
#: Calidad de esa recodificación (18 = prácticamente sin pérdida visible).
JOIN_CRF = 18
#: Preset: en una unión larga interesa que no eternice el proceso.
JOIN_PRESET = "veryfast"
#: A qué fps se igualan los trozos si no se puede sondear el primero.
JOIN_FALLBACK_FPS = 50.0
#: Frecuencia a la que se lleva el audio de cada trozo al igualarlos.
JOIN_SAMPLE_RATE = 48000
#: Ritmos «de verdad» a los que se ajusta una medida rara al igualar trozos. Un
#: efecto de velocidad deja tasas nominales como 38,67 fps (un valor que nadie ha
#: grabado nunca): igualar a eso convertiría el montaje en algo con un ritmo absurdo.
JOIN_FPS_LADDER: tuple[float, ...] = (24.0, 25.0, 30.0, 48.0, 50.0, 60.0, 100.0, 120.0)


def snap_fps(value) -> float | None:
    """Ajusta un fps medido al ritmo estándar más cercano (``None`` si no vale)."""
    try:
        medida = float(value)
    except (TypeError, ValueError):
        return None
    if medida <= 0:
        return None
    return min(JOIN_FPS_LADDER, key=lambda tasa: abs(tasa - medida))


def join_target_fps(values) -> float:
    """fps al que se igualan los trozos: el estándar más alto de todos ellos.

    Se mide cada trozo, se ajusta su ritmo al estándar más cercano y se toma el
    mayor: así manda la cámara (50 fps en AVCHD 1080i/p) y no el ritmo raro que deja
    un efecto de velocidad en un trozo concreto.
    """
    medidas = [tasa for tasa in (snap_fps(v) for v in values or ()) if tasa]
    return max(medidas) if medidas else JOIN_FALLBACK_FPS


def join_paths_ordered(paths) -> tuple[Path, ...]:
    """Rutas válidas, sin duplicados y **ordenadas por nombre**.

    Las cámaras trocean la grabación en ``00000.MTS``, ``00001.MTS``…, así que el
    orden alfabético es el orden real de grabación (el selector de ficheros no
    garantiza ninguno). Se descartan las rutas repetidas y las que no existen: es
    mejor unir las nueve buenas que fallar por una que no está.
    """
    vistas: dict[str, Path] = {}
    for ruta in paths or ():
        path = Path(str(ruta))
        if not path.is_file():
            continue
        vistas.setdefault(str(path).lower(), path)
    return tuple(sorted(vistas.values(), key=lambda p: p.name.lower()))


def join_command(ffmpeg: str, list_path, output) -> list[str]:
    """Une los trozos **copiando** el vídeo (rápido y sin pérdida).

    Es el primer intento, el que vale cuando todos los trozos salen de la misma
    cámara o del mismo proceso de la app con los mismos ajustes. El demuxer
    ``concat`` exige que **todos los ficheros tengan los mismos streams**: en cuanto
    a un trozo se le cambia el códec de audio (una rampa lo pasa a AAC y la cámara
    graba AC-3) o el tamaño, esto no puede funcionar y no hay opción de filtro que
    lo arregle — el fallo ocurre al leer, antes de los filtros. Para eso está
    :func:`join_normalize_command`.

    El audio sí se recodifica a AAC: las cámaras graban AC-3 y un MP4 con AAC es lo
    que abren todos los reproductores y lo que aceptan los filtros del resto de la
    app. Los subtítulos PGS de la cámara se descartan a propósito.
    """
    return [
        ffmpeg, "-y", "-hide_banner",
        "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "copy", "-c:a", "aac", "-b:a", JOIN_AUDIO_BITRATE,
        "-movflags", "+faststart",
        str(output),
    ]


def join_filter_complex(
    count: int,
    *,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    sample_rate: int = JOIN_SAMPLE_RATE,
) -> str:
    """Normaliza **cada trozo** y los pega con el filtro ``concat``.

    Se usa una entrada de ffmpeg por fichero (no el demuxer), así que da igual que
    los trozos tengan distinto códec de audio, tamaño o ritmo: cada uno se escala
    con relleno a un tamaño común, se pasa a fps constante y su audio se lleva a la
    misma frecuencia y a estéreo. Es la forma robusta de unir «trozo procesado +
    trozo crudo», que es justo lo que no aguanta el demuxer.

    El orden ``setpts`` **antes** de ``fps`` es el mismo que usa el Ensamblador en
    ``xfade``: al revés, ffmpeg 7.x deja la tasa en 1/0 y el filtrado falla.
    """
    total = max(1, int(count))
    ancho = int(width) if width else 1920
    alto = int(height) if height else 1080
    tasa = float(fps) if fps and fps > 0 else JOIN_FALLBACK_FPS
    partes: list[str] = []
    etiquetas: list[str] = []
    for indice in range(total):
        partes.append(
            f"[{indice}:v]scale={ancho}:{alto}:force_original_aspect_ratio=decrease,"
            f"pad={ancho}:{alto}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
            f"setpts=PTS-STARTPTS,fps={tasa:g},format=yuv420p[v{indice}]"
        )
        partes.append(
            f"[{indice}:a]aresample={int(sample_rate)},"
            "aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"asetpts=PTS-STARTPTS[a{indice}]"
        )
        etiquetas += [f"[v{indice}]", f"[a{indice}]"]
    partes.append(f"{''.join(etiquetas)}concat=n={total}:v=1:a=1[v][a]")
    return ";".join(partes)


def join_normalize_command(
    ffmpeg: str,
    paths,
    output,
    *,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    crf: int = JOIN_CRF,
    preset: str = JOIN_PRESET,
) -> list[str]:
    """Comando que une trozos **normalizándolos** (recodifica). Ver :func:`join_filter_complex`."""
    rutas = [Path(str(ruta)) for ruta in paths]
    argv = [ffmpeg, "-y", "-hide_banner"]
    for ruta in rutas:
        argv += ["-i", str(ruta)]
    argv += [
        "-filter_complex",
        join_filter_complex(len(rutas), width=width, height=height, fps=fps),
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", JOIN_AUDIO_BITRATE,
        "-movflags", "+faststart",
        str(output),
    ]
    return argv

#: FPS de interpolación ofrecidos (None = no tocar).
FPS_CHOICES: tuple[int, ...] = (24, 25, 30, 50, 60)

#: Escalas de escalado IA.
AI_SCALES: tuple[int, ...] = (2, 4)
AI_SCALE_DEFAULT = 2


def temperature_kelvin(level) -> int:
    """Kelvin de ``colortemperature`` para un nivel de calidez −100…+100.

    0 es el neutro (6500 K, lo que ya hacía el vídeo). Positivo = más cálido y
    negativo = más frío, **en el sentido del filtro**: por debajo de 6500 la imagen
    tira a naranja y por encima, a azul (medido en este equipo con un gris neutro).
    """
    try:
        valor = float(level)
    except (TypeError, ValueError):
        return TEMPERATURE_NEUTRAL_K
    valor = max(TEMPERATURE_MIN, min(TEMPERATURE_MAX, valor))
    objetivo = TEMPERATURE_WARM_K if valor >= 0 else TEMPERATURE_COOL_K
    recorrido = (objetivo - TEMPERATURE_NEUTRAL_K) * abs(valor) / 100.0
    return int(round(TEMPERATURE_NEUTRAL_K + recorrido))


@dataclass(frozen=True)
class ColorAdjust:
    """Ajustes de color: los del filtro ``eq`` más la calidez (temperatura)."""

    brightness: float = BRIGHTNESS_DEFAULT
    contrast: float = CONTRAST_DEFAULT
    saturation: float = SATURATION_DEFAULT
    gamma: float = GAMMA_DEFAULT
    temperature: float = TEMPERATURE_DEFAULT

    @property
    def is_identity(self) -> bool:
        return (
            abs(self.brightness - BRIGHTNESS_DEFAULT) < 1e-6
            and abs(self.contrast - CONTRAST_DEFAULT) < 1e-6
            and abs(self.saturation - SATURATION_DEFAULT) < 1e-6
            and abs(self.gamma - GAMMA_DEFAULT) < 1e-6
            and abs(self.temperature - TEMPERATURE_DEFAULT) < 1e-6
        )

    def eq_filter(self) -> str | None:
        if (
            abs(self.brightness - BRIGHTNESS_DEFAULT) < 1e-6
            and abs(self.contrast - CONTRAST_DEFAULT) < 1e-6
            and abs(self.saturation - SATURATION_DEFAULT) < 1e-6
            and abs(self.gamma - GAMMA_DEFAULT) < 1e-6
        ):
            return None
        return (
            f"eq=brightness={self.brightness:g}:contrast={self.contrast:g}:"
            f"saturation={self.saturation:g}:gamma={self.gamma:g}"
        )

    def temperature_filter(self) -> str | None:
        """Filtro ``colortemperature`` de la calidez, o ``None`` si está neutra."""
        if abs(self.temperature - TEMPERATURE_DEFAULT) < 1e-6:
            return None
        return f"colortemperature=temperature={temperature_kelvin(self.temperature)}"


def denoise_filter(level: str) -> str | None:
    return DENOISE_FILTERS.get((level or "off").lower())


def sharpen_filter(amount: float) -> str | None:
    if amount <= 1e-6:
        return None
    return f"unsharp=5:5:{amount:g}:5:5:0"


def fps_filter(fps: int | None) -> str | None:
    """Interpola a ``fps`` con ``minterpolate`` (mejor que duplicar fotogramas)."""
    if not fps:
        return None
    return (
        f"minterpolate=fps={int(fps)}:mi_mode=mci:mc_mode=aobmc:"
        "me_mode=bidir:vsbmc=1"
    )


# ---- Velocidad: cámara lenta / rápida ------------------------------


def clamp_speed(value) -> float:
    """Acota la velocidad a lo que tiene sentido (0.25x–2x; basura -> 1x)."""
    try:
        velocidad = float(value)
    except (TypeError, ValueError):
        return SPEED_DEFAULT
    if velocidad <= 0:
        # Dividir PTS entre 0 o entre un negativo daría un vídeo imposible: se
        # trata como «sin cambio», igual que un valor que no es un número.
        return SPEED_DEFAULT
    return min(max(SPEED_MIN, velocidad), SPEED_MAX)


def clamp_replay_times(value) -> int:
    """Acota cuántas veces se repite el momento (1–6; basura -> 3)."""
    try:
        veces = int(value)
    except (TypeError, ValueError):
        return REPLAY_TIMES_DEFAULT
    return min(max(REPLAY_MIN_TIMES, veces), REPLAY_MAX_TIMES)


def atempo_chain(speed) -> list[str]:
    """Filtros ``atempo`` para ``speed``, encadenados si ffmpeg no lo admite.

    ``atempo`` sólo acepta 0.5–2.0 por paso, así que 0.25x son dos ``atempo=0.5``
    (y un 4x serían dos ``atempo=2``). Sin esto, un vídeo a 0.25x con el audio
    copiado se desincronizaría desde el primer segundo.
    """
    factor = clamp_speed(speed)
    pasos: list[float] = []
    resto = factor
    while resto < ATEMPO_MIN - 1e-9:
        pasos.append(ATEMPO_MIN)
        resto /= ATEMPO_MIN
    while resto > ATEMPO_MAX + 1e-9:
        pasos.append(ATEMPO_MAX)
        resto /= ATEMPO_MAX
    pasos.append(round(resto, 6))
    return [f"atempo={paso:g}" for paso in pasos]


def audio_speed_filter(speed) -> str | None:
    """Cadena ``-af`` que mantiene el audio pegado al vídeo, o ``None`` a 1x."""
    if abs(clamp_speed(speed) - SPEED_DEFAULT) < 1e-6:
        return None
    return ",".join(atempo_chain(speed))


def smooth_slow_fps(speed, source_fps: float | None = None) -> int:
    """fps a interpolar para que la cámara lenta no vaya a saltos.

    Se interpola a ``fps de origen / velocidad`` (0.5x de 30 fps -> 60) y se acota
    con :data:`SMOOTH_MAX_FPS`: más allá no se aprecia y sólo cuesta tiempo.
    """
    try:
        origen = float(source_fps) if source_fps else SMOOTH_FALLBACK_FPS
    except (TypeError, ValueError):
        origen = SMOOTH_FALLBACK_FPS
    if origen <= 0:
        origen = SMOOTH_FALLBACK_FPS
    objetivo = round(origen / clamp_speed(speed))
    return int(min(SMOOTH_MAX_FPS, max(origen, objetivo)))


def speed_filter(
    speed, *, smooth: bool = False, source_fps: float | None = None
) -> str | None:
    """``setpts`` (y el interpolado previo) para cambiar la velocidad.

    ``smooth`` sólo se aplica a la cámara lenta: primero se *inventan* los
    fotogramas intermedios con ``minterpolate`` y luego se estira el tiempo con
    ``setpts``; así el movimiento sigue siendo continuo en vez de repetir el mismo
    fotograma. Es bastante más lento de procesar.
    """
    factor = clamp_speed(speed)
    if abs(factor - SPEED_DEFAULT) < 1e-6:
        return None
    partes: list[str] = []
    if smooth and factor < SPEED_DEFAULT:
        interpolado = fps_filter(smooth_slow_fps(factor, source_fps))
        if interpolado:
            partes.append(interpolado)
    partes.append(f"setpts=PTS/{factor:g}")
    return ",".join(partes)


def build_filters(
    adjust: ColorAdjust | None = None,
    denoise: str = "off",
    sharpen: float = SHARPEN_DEFAULT,
    fps: int | None = None,
    speed=SPEED_DEFAULT,
    smooth: bool = False,
    source_fps: float | None = None,
    zoom=ZOOM_DEFAULT,
    zoom_size: tuple[int, int] | None = None,
) -> str:
    """Cadena ``-vf`` con los filtros pedidos (vacía si no hay ninguno).

    El interpolado de FPS del usuario va **antes** del ``setpts`` de la velocidad:
    al revés, ffmpeg puede quedarse con una tasa de 1/0 y rechazar el filtrado
    (es el mismo caso que documenta ``core.video_assembler``). El zoom va el
    primero: recortar antes deja menos píxeles para el resto de la cadena.
    """
    partes: list[str] = []
    acercar = zoom_filter(zoom, zoom_size)
    if acercar:
        partes.append(acercar)
    if adjust is not None:
        eq = adjust.eq_filter()
        if eq:
            partes.append(eq)
        temperatura = adjust.temperature_filter()
        if temperatura:
            partes.append(temperatura)
    den = denoise_filter(denoise)
    if den:
        partes.append(den)
    nit = sharpen_filter(sharpen)
    if nit:
        partes.append(nit)
    interp = fps_filter(fps)
    if interp:
        partes.append(interp)
    velocidad = speed_filter(speed, smooth=smooth, source_fps=source_fps)
    if velocidad:
        partes.append(velocidad)
    return ",".join(partes)


# ---- Rótulos (texto sobre el vídeo) --------------------------------

#: Segundos que dura cada rótulo si la línea no dice otra cosa.
ROLULO_DEFAULT_SECONDS = 2.5
#: Tamaño de letra por defecto (px, con la resolución real del vídeo).
ROLULO_FONTSIZE = 34
#: Ejemplo que se muestra en la UI: una línea por rótulo, ``minuto:segundo texto``.
ROLULO_EXAMPLE = (
    "0:00 Balonmano · Jornada 5\n"
    "0:12 Gol de Álex · 12-10\n"
    "1:05 Parada de Unai"
)


@dataclass(frozen=True)
class Rolulo:
    """Texto que aparece sobre el vídeo a partir del segundo ``t_s``."""

    t_s: float
    texto: str


def parse_timestamp(value: str) -> float:
    """``"0:12"`` → 12.0; admite ``"1:02:03"``, ``"95"`` y ``"95.5"``.

    Lanza ``ValueError`` si no es una marca de tiempo, para que la UI pueda decir
    *qué* línea está mal en vez de tragarse el rótulo en silencio.
    """
    texto = (value or "").strip().replace(",", ".")
    if not texto:
        raise ValueError("falta la marca de tiempo")
    partes = texto.split(":")
    if len(partes) > 3:
        raise ValueError(f"«{value}» no es una marca de tiempo")
    try:
        numeros = [float(parte) for parte in partes]
    except ValueError as exc:
        raise ValueError(f"«{value}» no es una marca de tiempo") from exc
    segundos = 0.0
    for numero in numeros:
        segundos = segundos * 60 + numero
    return segundos


def parse_rolulos(
    text: str,
    *,
    default_seconds: float = ROLULO_DEFAULT_SECONDS,
    duration: float | None = None,
) -> list[Rolulo]:
    """Convierte el texto de la UI (``minuto:segundo frase``) en rótulos.

    Una línea por rótulo. Se ignoran las líneas vacías y las que empiezan por
    ``#`` (comentarios), y se ordenan por tiempo. Si se conoce la ``duración`` del
    vídeo, un rótulo no se sale de él: su final se recorta al final del vídeo.
    """
    rotulos: list[Rolulo] = []
    for numero, linea in enumerate((text or "").splitlines(), start=1):
        limpia = linea.strip()
        if not limpia or limpia.startswith("#"):
            continue
        cabeza, _, frase = limpia.partition(" ")
        frase = frase.strip()
        if not frase:
            raise ValueError(
                f"Línea {numero}: falta el texto del rótulo "
                f"(formato «minuto:segundo frase», p. ej. «0:12 Gol de Álex»)."
            )
        try:
            t_s = parse_timestamp(cabeza)
        except ValueError as exc:
            raise ValueError(f"Línea {numero}: {exc}.") from exc
        if duration is not None and t_s >= float(duration):
            # Fuera del vídeo: no se vería nunca. Mejor avisar que ignorarlo.
            raise ValueError(
                f"Línea {numero}: el rótulo empieza en {t_s:g}s y el vídeo dura "
                f"{float(duration):g}s."
            )
        rotulos.append(Rolulo(round(t_s, 3), frase))
    if not rotulos:
        raise ValueError("Escribe al menos un rótulo (una línea por rótulo).")
    return sorted(rotulos, key=lambda r: r.t_s)


def rolulos_srt(
    rotulos: Sequence[Rolulo], *, default_seconds: float = ROLULO_DEFAULT_SECONDS
) -> str:
    """SRT de los rótulos (lo quema ``VideoAssembler`` con su estilo ASS)."""
    return build_srt(
        [(r.t_s, r.t_s + max(0.2, float(default_seconds)), r.texto) for r in rotulos]
    )


# ---- Repeticiones de un momento (replay) ---------------------------


def clamp_zoom(value) -> float:
    """Acota el zoom (1x–3x; basura -> 1x, que es «sin zoom»)."""
    try:
        factor = float(value)
    except (TypeError, ValueError):
        return ZOOM_DEFAULT
    if factor <= 0:
        return ZOOM_DEFAULT
    return min(max(ZOOM_MIN, factor), ZOOM_MAX)


def zoom_filter(factor, size: tuple[int, int] | None = None) -> str | None:
    """Acerca la imagen recortando el centro y reescalando (``None`` sin zoom).

    Se hace con ``crop``+``scale`` y no con ``zoompan``: ``crop`` recorta por el
    centro por defecto y no necesita fps, duraciones ni expresiones temporales, así
    que el mismo filtro vale para el vídeo entero y para cada tramo de una repetición.

    ``size`` (ancho, alto del vídeo) es opcional pero recomendable: con él el
    resultado vuelve **exactamente** al tamaño original. Sin él se deduce de las
    dimensiones recortadas y la resolución puede bailar un par de píxeles (medido:
    un zoom 1,5x sobre 640x360 daba 638x360), lo que rompería un ``concat``
    posterior en el Ensamblador, que exige parámetros idénticos. Todo se redondea a
    par (``trunc(.../2)*2``): H.264 con ``yuv420p`` rechaza dimensiones impares.
    """
    valor = clamp_zoom(factor)
    if abs(valor - ZOOM_DEFAULT) < 1e-6:
        return None
    recorte = f"crop=trunc(iw/{valor:g}/2)*2:trunc(ih/{valor:g}/2)*2"
    if size and size[0] and size[1]:
        return f"{recorte},scale={int(size[0])}:{int(size[1])}"
    return f"{recorte},scale=trunc(iw*{valor:g}/2)*2:trunc(ih*{valor:g}/2)*2"


@dataclass(frozen=True)
class TimelineSegment:
    """Tramo del vídeo (``start``–``end``) y a qué velocidad se reproduce."""

    start: float
    end: float
    speed: float = SPEED_DEFAULT
    #: Interpolar los fotogramas del tramo (sólo tiene sentido si va lento).
    smooth: bool = False

    @property
    def is_slow(self) -> bool:
        return self.speed < SPEED_DEFAULT


@dataclass(frozen=True)
class TimelineGraph:
    """Grafo ``-filter_complex`` de una línea de tiempo y sus salidas.

    Lo comparten el *replay* (el mismo momento varias veces) y la **rampa** de
    velocidad (el vídeo entero con un tramo ralentizado).
    """

    graph: str
    segments: tuple[TimelineSegment, ...] = ()
    zoom: float = ZOOM_DEFAULT
    slow_last: bool = False
    audio_label: str | None = "[a]"
    video_label: str = "[v]"

    @property
    def times(self) -> int:
        """Cuántos tramos monta el grafo (las «vueltas» de un replay)."""
        return len(self.segments)

    @property
    def maps(self) -> tuple[str, ...]:
        """Etiquetas a mapear en la salida (vídeo y, si la hay, audio)."""
        if self.audio_label is None:
            return (self.video_label,)
        return (self.video_label, self.audio_label)


def timeline_filter_complex(
    segments: Sequence[TimelineSegment],
    *,
    has_audio: bool = True,
    source_fps: float | None = None,
    zoom=ZOOM_DEFAULT,
    zoom_size: tuple[int, int] | None = None,
) -> TimelineGraph:
    """Encadena ``segments`` (recorte + velocidad) en **un solo pase** de ffmpeg.

    Cada tramo se recorta con ``trim``/``atrim``, se le aplica su velocidad con
    ``setpts`` (y ``atempo`` en el audio, para que no se desincronice) y el
    resultado se pega con ``concat``. El zoom, si lo hay, se aplica **una vez** al
    resultado del ``concat``: recortar después cuesta una operación y no una por
    tramo.
    """
    tramos = tuple(segments)
    if not tramos:
        raise ValueError("La línea de tiempo no tiene ningún tramo.")

    partes: list[str] = []
    etiquetas: list[str] = []
    for indice, tramo in enumerate(tramos):
        factor = clamp_speed(tramo.speed)
        suave = ""
        if tramo.smooth and factor < SPEED_DEFAULT:
            interpolado = fps_filter(smooth_slow_fps(factor, source_fps))
            suave = f"{interpolado}," if interpolado else ""
        partes.append(
            f"[0:v]trim=start={tramo.start:g}:end={tramo.end:g},"
            f"{suave}setpts=(PTS-STARTPTS)/{factor:g}[v{indice}]"
        )
        etiquetas.append(f"[v{indice}]")
        if has_audio:
            cadena = (
                f"[0:a]atrim=start={tramo.start:g}:end={tramo.end:g},"
                "asetpts=PTS-STARTPTS"
            )
            audio_speed = audio_speed_filter(factor)
            if audio_speed:
                cadena += f",{audio_speed}"
            partes.append(f"{cadena}[a{indice}]")
            etiquetas.append(f"[a{indice}]")

    # ``n`` es el número de *segmentos*, no de pads: con audio cada segmento
    # aporta dos (imagen y sonido). Poner n=6 para 3 repeticiones con audio da
    # «Cannot find a matching stream for unlabeled input pad concat» (ffmpeg 7.1).
    salidas = "[vv][a]" if has_audio else "[vv]"
    partes.append(
        f"{''.join(etiquetas)}concat=n={len(tramos)}:v=1:a={1 if has_audio else 0}"
        f"{salidas}"
    )

    zoom_cadena = zoom_filter(zoom, zoom_size)
    if zoom_cadena:
        partes.append(f"[vv]{zoom_cadena}[v]")
    else:
        partes.append("[vv]null[v]")

    return TimelineGraph(
        graph=";".join(partes),
        segments=tramos,
        zoom=clamp_zoom(zoom),
        slow_last=bool(tramos and tramos[-1].is_slow),
        audio_label="[a]" if has_audio else None,
    )


def replay_filter_complex(
    start,
    end,
    times=REPLAY_TIMES_DEFAULT,
    *,
    slow_last: bool = True,
    slow_speed=REPLAY_SLOW_SPEED,
    has_audio: bool = True,
    smooth: bool = False,
    source_fps: float | None = None,
    zoom=ZOOM_DEFAULT,
    zoom_size: tuple[int, int] | None = None,
) -> TimelineGraph:
    """Repite el fragmento ``[start, end]`` ``times`` veces, en un solo pase.

    La última repetición puede ir en cámara lenta (:data:`REPLAY_SLOW_SPEED`), que
    es lo que convierte el momento en «repetición de jugada». El audio acompaña al
    vídeo con ``atempo`` para no desincronizarse; si el vídeo no trae audio se
    monta sólo la parte de imagen (mapear ``[0:a]`` en un fichero mudo haría fallar
    el filtrado).

    Lanza ``ValueError`` si el fragmento no es válido (fin ≤ inicio), para que la
    UI lo pueda explicar en la barra de estado en vez de soltar el error crudo.
    """
    try:
        inicio, fin = float(start), float(end)
    except (TypeError, ValueError) as exc:
        raise ValueError("El inicio y el fin del fragmento deben ser números.") from exc
    if fin <= inicio:
        raise ValueError("El fin del fragmento debe ser mayor que su inicio.")

    veces = clamp_replay_times(times)
    velocidad = clamp_speed(slow_speed)
    lenta = abs(velocidad - SPEED_DEFAULT) > 1e-6
    tramos = [
        TimelineSegment(
            inicio,
            fin,
            velocidad if (slow_last and lenta and indice == veces - 1) else SPEED_DEFAULT,
            smooth=smooth and slow_last and lenta and indice == veces - 1,
        )
        for indice in range(veces)
    ]
    return timeline_filter_complex(
        tramos,
        has_audio=has_audio,
        source_fps=source_fps,
        zoom=zoom,
        zoom_size=zoom_size,
    )


def speed_ramp_filter_complex(
    duration,
    start,
    end,
    *,
    slow_speed=REPLAY_SLOW_SPEED,
    has_audio: bool = True,
    smooth: bool = False,
    source_fps: float | None = None,
    zoom=ZOOM_DEFAULT,
    zoom_size: tuple[int, int] | None = None,
) -> TimelineGraph:
    """Ralentiza **sólo** el tramo ``[start, end]`` y deja el resto igual.

    Es la «rampa» de las retransmisiones: el partido sigue a velocidad normal y, al
    llegar al momento, baja la velocidad para verlo con detalle y luego vuelve.
    """
    try:
        total = float(duration)
        inicio, fin = float(start), float(end)
    except (TypeError, ValueError) as exc:
        raise ValueError("La duración y el tramo deben ser números.") from exc
    if total <= 0:
        raise ValueError("No se conoce la duración del vídeo.")
    if fin <= inicio:
        raise ValueError("El fin del tramo debe ser mayor que su inicio.")
    if inicio < 0:
        raise ValueError("El tramo no puede empezar antes del segundo 0.")

    velocidad = clamp_speed(slow_speed)
    if abs(velocidad - SPEED_DEFAULT) < 1e-6:
        raise ValueError("La velocidad del tramo lento ya es 1x: no hay rampa.")

    tramos: list[TimelineSegment] = []
    if inicio > 1e-3:
        tramos.append(TimelineSegment(0.0, min(inicio, total)))
    tramos.append(TimelineSegment(inicio, min(fin, total), velocidad, smooth=smooth))
    if fin < total - 1e-3:
        tramos.append(TimelineSegment(fin, total))
    return timeline_filter_complex(
        tramos,
        has_audio=has_audio,
        source_fps=source_fps,
        zoom=zoom,
        zoom_size=zoom_size,
    )
