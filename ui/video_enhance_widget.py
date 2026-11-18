"""Pestaña **Vídeo IA**: reescala, ajusta el color, cambia los fotogramas y la velocidad.

- **Color/efectos (ffmpeg)**: brillo, contraste, saturación, gamma, reduce-ruido,
  nitidez e interpolación de FPS (``minterpolate``). Rápido y siempre disponible.
- **Cámara lenta y repeticiones**: velocidad 0.25x–2x (con el audio reajustado con
  ``atempo`` para que no se desincronice) y *replay* de un momento: el fragmento
  repetido N veces con la última vuelta en cámara lenta. Pensado para volver a ver
  un gol o una parada de un partido.
- **Escalado con IA (Real-ESRGAN)**: extrae los fotogramas, los reescala con el
  worker de ``venv-photo`` (fotograma a fotograma, cargando el modelo una vez) y
  vuelve a montar el vídeo con el audio original.

La construcción de filtros y del grafo del replay vive en
:mod:`services.video_enhance_params`.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import threading
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:
    import imageio_ffmpeg
except ImportError:  # pragma: no cover - dependencia declarada en requirements.txt
    imageio_ffmpeg = None

from core.frame_extractor import FrameExtractor
from core.paths import OUTPUT, PROJECT_ROOT
from core.subprocess_utils import no_window_kwargs
from core.video_assembler import VideoAssembler
from engines.base_engine import CANCEL_MESSAGE, EngineError
from services import assembler_params as asp
from services import video_enhance_params as vp
from services import video_understanding as vu
from ui.task_widget import TaskWidget
from ui.workers import ThreadWorker

logger = logging.getLogger(__name__)

VIDEO_PYTHON = "venv-photo/Scripts/python.exe"
VIDEO_WORKER = "tools/video_upscale_worker.py"
VIDEO_MODELS_DIR = "tools/models"

#: Carpeta de las grabaciones unidas. Va en ``projects/``, **fuera** de ``output/``:
#: la retención de salidas borra por antigüedad y tamaño, y ahí una grabación de
#: 10 GB no debe desaparecer.
RECORDINGS_DIR = PROJECT_ROOT / "projects" / "grabaciones"

FFMPEG_TIMEOUT = 3600
UPSCALE_TIMEOUT = 7200


def _ffmpeg_exe() -> str:
    if imageio_ffmpeg is None:
        raise EngineError("imageio-ffmpeg no está instalado.")
    return imageio_ffmpeg.get_ffmpeg_exe()


def _duracion(video) -> float | None:
    """Duración del vídeo en segundos, o ``None`` si no se puede leer (es barato)."""
    try:
        return VideoAssembler().probe(video).get("duration")
    except EngineError:
        return None


def _run_ffmpeg(
    argv: list[str],
    etiqueta: str,
    timeout: float = FFMPEG_TIMEOUT,
    *,
    duration: float | None = None,
    on_percent=None,
    stage: tuple[float, float] = (0.0, 100.0),
    cancel_check=None,
):
    """Ejecuta ffmpeg y devuelve un ``CompletedProcess``.

    Con ``duration`` y ``on_percent`` se le pide a ffmpeg el avance **real**
    (``-progress``) y se traduce a porcentaje: es la diferencia entre una barra que
    avanza sola y una etiqueta clavada en «Aplicando filtros…» durante minutos. Con
    ``cancel_check`` el proceso se mata a mitad (antes sólo se cancelaba entre pasos).
    """
    comando = list(argv)
    con_progreso = duration is not None and on_percent is not None
    if con_progreso:
        # Justo antes del fichero de salida, que es el último argumento.
        comando[-1:-1] = ["-progress", "pipe:1", "-nostats"]
    try:
        proceso = subprocess.Popen(
            comando,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if con_progreso else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **no_window_kwargs(),
        )
    except OSError as exc:
        raise EngineError(f"No se pudo ejecutar ffmpeg: {exc}") from exc

    # ffmpeg escribe su log en stderr mientras nosotros leemos el avance por
    # stdout. Si stderr no se drena a la vez, su tubería se llena y ffmpeg —y
    # nosotros con él— nos quedamos esperando: bloqueo mutuo. Un hilo la vacía.
    bolsa_error: list[str] = []

    def _drenar_error() -> None:
        if proceso.stderr is not None:
            bolsa_error.append(proceso.stderr.read())

    hilo_error = threading.Thread(target=_drenar_error, daemon=True)
    hilo_error.start()

    try:
        if con_progreso and proceso.stdout is not None:
            for linea in proceso.stdout:
                if cancel_check is not None and cancel_check():
                    proceso.kill()
                    proceso.wait()
                    raise EngineError(CANCEL_MESSAGE)
                segundos = vp.parse_out_time(linea)
                if segundos is not None:
                    on_percent(int(vp.progress_percent(segundos, duration, stage=stage)))
        try:
            proceso.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            proceso.kill()
            proceso.wait()
            raise EngineError(
                f"{etiqueta} (ffmpeg no terminó en {timeout:.0f}s)."
            ) from exc
        hilo_error.join(timeout=5)
        error = bolsa_error[0] if bolsa_error else ""
        return subprocess.CompletedProcess(argv, proceso.returncode, "", error)
    finally:
        for flujo in (proceso.stdout, proceso.stderr):
            if flujo is not None:
                flujo.close()


class FfmpegEnhanceWorker(ThreadWorker):
    """Aplica los filtros de ffmpeg (color/efectos/FPS/velocidad) y guarda el vídeo.

    Con ``audio_filter`` (p. ej. ``atempo=0.5,atempo=0.5``) el audio se reajusta y
    se **recodifica**: copiarlo dejaría el sonido desincronizado en cuanto se toca
    la velocidad del vídeo.
    """

    def __init__(
        self,
        video: Path,
        output: Path,
        filters: str,
        crf: int,
        parent=None,
        audio_filter: str | None = None,
        encoder: str = vp.ENCODER_CPU,
        audio_codec: str = "copy",
    ) -> None:
        super().__init__(parent)
        self._video = video
        self._output = output
        self._filters = filters
        self._crf = crf
        self._audio_filter = audio_filter
        self._encoder = encoder
        self._audio_codec = audio_codec

    def work(self) -> Path:
        ffmpeg = _ffmpeg_exe()
        argv = [ffmpeg, "-y", "-i", str(self._video)]
        if self._filters:
            argv += ["-vf", self._filters]
        if self._audio_filter:
            argv += ["-af", self._audio_filter]
        argv += vp.encoder_args(self._encoder, self._crf)
        if self._audio_filter:
            argv += ["-c:a", "aac"]
        elif self._audio_codec == "aac":
            # Comprimir: el AC-3 de las cámaras pasa a AAC (más compatible y más
            # pequeño) a un bitrate de sobra para lo que graba una cámara.
            argv += ["-c:a", "aac", "-b:a", vp.COMPRESSION_AUDIO_BITRATE]
        else:
            argv += ["-c:a", "copy"]
        argv += ["-movflags", "+faststart", str(self._output)]
        mensaje = "Aplicando filtros con ffmpeg…"
        self.progress.emit(5, mensaje)
        proc = _run_ffmpeg(
            argv,
            "ffmpeg falló",
            duration=_duracion(self._video),
            on_percent=lambda p: self.progress.emit(p, mensaje),
            stage=(5.0, 99.0),
            cancel_check=self.isInterruptionRequested,
        )
        if proc.returncode != 0 or not self._output.is_file():
            cola = (proc.stderr or "").strip()[-600:]
            raise EngineError(f"ffmpeg falló. {cola}")
        self.progress.emit(100, "Vídeo listo.")
        return self._output


class ReplayWorker(ThreadWorker):
    """Monta el replay de un momento: N repeticiones y la última en cámara lenta.

    Todo va en **un solo pase** de ffmpeg (``-filter_complex``): recortar el
    fragmento, repetirlo y ralentizar la última vuelta sin escribir ficheros
    intermedios ni recodificar tres veces el mismo vídeo.
    """

    def __init__(
        self,
        video: Path,
        output: Path,
        graph: vp.ReplayGraph,
        crf: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._video = video
        self._output = output
        self._graph = graph
        self._crf = crf

    def work(self) -> Path:
        ffmpeg = _ffmpeg_exe()
        argv = [
            ffmpeg, "-y", "-i", str(self._video),
            "-filter_complex", self._graph.graph,
        ]
        for etiqueta in self._graph.maps:
            argv += ["-map", etiqueta]
        argv += [
            "-c:v", "libx264", "-crf", str(self._crf), "-preset", "medium",
            "-pix_fmt", "yuv420p",
        ]
        # Un vídeo mudo no tiene etiqueta ``[a]`` que mapear: sin audio no se
        # declara códec de audio, sólo se pide no incluirlo.
        argv += ["-an"] if self._graph.audio_label is None else ["-c:a", "aac"]
        argv += [str(self._output)]
        mensaje = f"Montando la repetición ({self._graph.times} veces)…"
        self.progress.emit(5, mensaje)
        proc = _run_ffmpeg(
            argv,
            "El replay falló",
            duration=_duracion(self._video),
            on_percent=lambda p: self.progress.emit(p, mensaje),
            stage=(5.0, 99.0),
            cancel_check=self.isInterruptionRequested,
        )
        if proc.returncode != 0 or not self._output.is_file():
            cola = (proc.stderr or "").strip()[-600:]
            raise EngineError(f"El replay falló. {cola}")
        self.progress.emit(100, "Repetición lista.")
        return self._output


class JoinWorker(ThreadWorker):
    """Une grabaciones troceadas (los ``.MTS`` de una cámara) en un solo MP4.

    Primero intenta **copiar** los streams: leer y pegar diez trozos de 1080p tarda
    segundos, no horas. Si los trozos no coinciden —porque a alguno se le puso una
    rampa, una repetición o una cámara lenta, que cambian el ritmo, o un escalado,
    que cambia el tamaño—, reintenta **igualando** tamaño y fps, que es lo que
    permite el flujo de «proceso cada trozo y los uno al final».
    """

    def __init__(self, paths, output: Path, *, reencode: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._paths = tuple(paths)
        self._output = Path(output)
        self._reencode = bool(reencode)

    def _probe_all(self) -> list[dict]:
        """Sonda cada trozo (barato: ``ffmpeg -i``) para saber tamaño y ritmo."""
        datos: list[dict] = []
        for path in self._paths:
            try:
                datos.append(VideoAssembler().probe(path))
            except EngineError as exc:
                logger.debug("Unir: no se pudo sondear %s: %s", path.name, exc)
                datos.append({})
        return datos

    def _argv(
        self, ffmpeg: str, lista: Path, reencode: bool, info: list[dict]
    ) -> list[str]:
        if not reencode:
            return vp.join_command(ffmpeg, lista, self._output)
        # Igualar: una entrada por trozo y el filtro ``concat`` (el demuxer no puede
        # pegar streams de distinto códec de audio, y eso pasa en cuanto un trozo ha
        # pasado por un efecto de velocidad). El tamaño sale del primer trozo y el
        # ritmo, del estándar más alto de todos: manda la cámara, no el fps raro que
        # deja una rampa.
        primero = info[0] if info else {}
        return vp.join_normalize_command(
            ffmpeg,
            self._paths,
            self._output,
            width=primero.get("width"),
            height=primero.get("height"),
            fps=vp.join_target_fps([datos.get("fps") for datos in info]),
        )

    def work(self) -> Path:
        ffmpeg = _ffmpeg_exe()
        self._output.parent.mkdir(parents=True, exist_ok=True)
        info = self._probe_all()

        with tempfile.TemporaryDirectory(prefix="imperator_unir_") as tmp:
            lista = Path(tmp) / "lista.txt"
            lista.write_text(
                VideoAssembler.concat_list_text(self._paths), encoding="utf-8"
            )
            duracion_total = sum(float(d.get("duration") or 0.0) for d in info) or None
            mensaje = f"Uniendo {len(self._paths)} grabaciones…"
            self.progress.emit(5, mensaje)
            proc = _run_ffmpeg(
                self._argv(ffmpeg, lista, self._reencode, info),
                "No se pudieron unir las grabaciones",
                timeout=vp.JOIN_REENCODE_TIMEOUT if self._reencode else vp.JOIN_TIMEOUT,
                duration=duracion_total,
                on_percent=lambda p: self.progress.emit(p, mensaje),
                stage=(5.0, 99.0),
                cancel_check=self.isInterruptionRequested,
            )
            if (proc.returncode != 0 or not self._output.is_file()) and not self._reencode:
                mensaje = "Los trozos no coinciden: igualando tamaño y fps…"
                self.progress.emit(20, mensaje)
                self._output.unlink(missing_ok=True)
                proc = _run_ffmpeg(
                    self._argv(ffmpeg, lista, True, info),
                    "No se pudieron unir las grabaciones",
                    timeout=vp.JOIN_REENCODE_TIMEOUT,
                    duration=duracion_total,
                    on_percent=lambda p: self.progress.emit(p, mensaje),
                    stage=(20.0, 99.0),
                    cancel_check=self.isInterruptionRequested,
                )
        if proc.returncode != 0 or not self._output.is_file():
            cola = (proc.stderr or "").strip()[-600:]
            raise EngineError(f"No se pudieron unir las grabaciones. {cola}")
        self.progress.emit(100, "Grabaciones unidas.")
        return self._output


class BatchCompressWorker(ThreadWorker):
    """Comprime varios vídeos seguidos (un trozo detrás de otro).

    Va **de uno en uno** a propósito: dos codificaciones a la vez se pelean por la
    misma CPU/GPU y por el mismo disco (hablamos de decenas de GB de lectura), así
    que en paralelo se tarda más y se calienta más. Se puede cancelar entre trozos.

    Un trozo dañado **no tira el lote**: se apunta en el resumen y se sigue con los
    demás; solo falla del todo si fallan todos.
    """

    def __init__(self, trabajos, quality: int, encoder: str, parent=None) -> None:
        super().__init__(parent)
        #: Pares ``(origen, destino)`` ya decididos por la UI.
        self._trabajos = tuple(trabajos)
        self._quality = int(quality)
        self._encoder = encoder

    def work(self) -> str:
        total = len(self._trabajos)
        entrada_mb = 0.0
        salida_mb = 0.0
        fallos: list[str] = []
        for indice, (origen, destino) in enumerate(self._trabajos):
            if self.isInterruptionRequested():
                logger.info("Lote de compresión cancelado en %d/%d.", indice, total)
                break
            try:
                entrada_mb += Path(origen).stat().st_size / 1024**2
            except OSError:
                pass
            self.progress.emit(
                vp.batch_progress(indice, 0, total),
                f"Comprimiendo {indice + 1}/{total} · {Path(origen).name}",
            )
            try:
                interno = FfmpegEnhanceWorker(
                    Path(origen), Path(destino), "", self._quality,
                    encoder=self._encoder, audio_codec="aac",
                )
                # El avance de cada trozo se traduce a su franja del total del lote:
                # así la barra avanza dentro del trozo y no a saltos de 10 en 10.
                interno.progress.connect(
                    lambda p, _msg, i=indice, nombre=Path(origen).name: self.progress.emit(
                        vp.batch_progress(i, p, total),
                        f"Comprimiendo {i + 1}/{total} · {nombre}",
                    )
                )
                interno.work()
                salida_mb += Path(destino).stat().st_size / 1024**2
            except EngineError as exc:
                logger.warning("Lote: falló %s: %s", Path(origen).name, exc)
                fallos.append(Path(origen).name)
            self.progress.emit(
                vp.batch_progress(indice, 100, total),
                f"{indice + 1}/{total} trozos hechos",
            )
        hechos = total - len(fallos)
        if hechos == 0:
            raise EngineError(
                "No se pudo comprimir ninguno de los trozos: " + ", ".join(fallos)
            )
        # Sin flechas ni símbolos fuera de cp1252: este texto puede acabar en la
        # consola de Windows (que no los sabe imprimir) además de en la barra.
        resumen = (
            f"{hechos}/{total} comprimidos · "
            f"{entrada_mb / 1024:.2f} GB -> {salida_mb / 1024:.2f} GB"
        )
        if fallos:
            resumen += " · fallaron: " + ", ".join(fallos)
        return resumen


class RoluloWorker(ThreadWorker):
    """Quema rótulos (minuto, marcador, nombres) sobre el vídeo.

    Reutiliza el ASS del Ensamblador (``VideoAssembler._build_ass``) y su filtro
    ``subtitles``: mismos estilos (posición, tamaño, color, contorno) y sin depender
    de rutas de fuentes de Windows. El vídeo se recodifica; el audio se copia.
    """

    def __init__(
        self,
        video: Path,
        output: Path,
        srt_text: str,
        style: dict,
        crf: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._video = video
        self._output = output
        self._srt_text = srt_text
        self._style = style
        self._crf = crf

    def work(self) -> Path:
        ffmpeg = _ffmpeg_exe()
        info = VideoAssembler().probe(self._video)
        with tempfile.TemporaryDirectory(prefix="imperator_rotulos_") as tmp:
            srt = Path(tmp) / "rotulos.srt"
            srt.write_text(self._srt_text, encoding="utf-8")
            ass = VideoAssembler()._build_ass(
                srt,
                self._style,
                video_width=info.get("width"),
                video_height=info.get("height"),
            )
            argv = [
                ffmpeg, "-y", "-i", str(self._video),
                "-vf", VideoAssembler.subtitles_filter(ass),
                "-c:v", "libx264", "-crf", str(self._crf), "-preset", "medium",
                "-pix_fmt", "yuv420p", "-c:a", "copy", str(self._output),
            ]
            mensaje = "Quemando los rótulos…"
            self.progress.emit(5, mensaje)
            proc = _run_ffmpeg(
                argv,
                "No se pudieron quemar los rótulos",
                duration=info.get("duration"),
                on_percent=lambda p: self.progress.emit(p, mensaje),
                stage=(5.0, 99.0),
                cancel_check=self.isInterruptionRequested,
            )
        if proc.returncode != 0 or not self._output.is_file():
            cola = (proc.stderr or "").strip()[-600:]
            raise EngineError(f"No se pudieron quemar los rótulos. {cola}")
        self.progress.emit(100, "Rótulos listos.")
        return self._output


class AiUpscaleWorker(ThreadWorker):
    """Extrae fotogramas, los reescala con Real-ESRGAN y remonta el vídeo."""

    def __init__(
        self, video: Path, output: Path, scale: int,
        python: Path, worker: Path, model_dir: Path, parent=None,
    ) -> None:
        super().__init__(parent)
        self._video = video
        self._output = output
        self._scale = scale
        self._python = python
        self._worker = worker
        self._model_dir = model_dir

    def _check_cancel(self) -> None:
        if self.isInterruptionRequested():
            raise EngineError(CANCEL_MESSAGE)

    def work(self) -> Path:
        extractor = FrameExtractor()
        duracion = extractor.duration(self._video)
        if not duracion or duracion <= 0:
            raise EngineError("No se pudo leer la duración del vídeo.")
        fps = self._probe_fps()
        total = max(1, int(round(duracion * fps)))

        with tempfile.TemporaryDirectory(prefix="imperator_vup_") as tmp:
            entrada = Path(tmp) / "in"
            salida = Path(tmp) / "out"
            entrada.mkdir()
            salida.mkdir()
            for i in range(total):
                self._check_cancel()
                extractor.extract_frame(self._video, i / fps, entrada / f"{i:05d}.png")
                self.progress.emit(int((i + 1) / total * 50), f"Extrayendo fotogramas {i + 1}/{total}…")

            self.progress.emit(55, "Reescalando con IA (puede tardar)…")
            argv = [
                str(self._python), str(self._worker),
                "--frames-in", str(entrada), "--frames-out", str(salida),
                "--scale", str(self._scale), "--model-dir", str(self._model_dir),
            ]
            try:
                proc = subprocess.run(
                    argv, capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=UPSCALE_TIMEOUT, **no_window_kwargs(),
                )
            except subprocess.TimeoutExpired as exc:
                raise EngineError(f"El escalado IA no terminó en {UPSCALE_TIMEOUT}s.") from exc
            except OSError as exc:
                raise EngineError(f"No se pudo ejecutar el worker de vídeo: {exc}") from exc
            detalle = self._last_json(proc.stdout)
            if proc.returncode != 0 or not detalle.get("ok", False):
                motivo = detalle.get("error") or (proc.stderr or "").strip()[-600:]
                raise EngineError(f"Escalado IA fallido. {motivo}")

            self.progress.emit(90, "Montando el vídeo final…")
            self._check_cancel()
            self._assemble(entrada, salida, fps, total)
        self.progress.emit(100, "Vídeo listo.")
        return self._output

    def _probe_fps(self) -> float:
        try:
            fps = VideoAssembler().probe(self._video).get("fps")
            if fps:
                return float(fps)
        except Exception as exc:  # noqa: BLE001 - sonda opcional
            logger.debug("No se pudo sondear el fps: %r", exc)
        return 30.0

    def _assemble(self, entrada: Path, salida: Path, fps: float, total: int) -> None:
        if imageio_ffmpeg is None:
            raise EngineError("imageio-ffmpeg no está instalado.")
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        patron = str(salida / "%05d.png")
        argv = [
            ffmpeg, "-y",
            "-framerate", f"{fps:g}", "-start_number", "0", "-i", patron,
            "-i", str(self._video),
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "copy", "-shortest", str(self._output),
        ]
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=FFMPEG_TIMEOUT, **no_window_kwargs(),
            )
        except subprocess.TimeoutExpired as exc:
            raise EngineError(f"ffmpeg no terminó montando el vídeo en {FFMPEG_TIMEOUT}s.") from exc
        except OSError as exc:
            raise EngineError(f"No se pudo ejecutar ffmpeg: {exc}") from exc
        if proc.returncode != 0 or not self._output.is_file():
            cola = (proc.stderr or "").strip()[-600:]
            raise EngineError(f"ffmpeg falló al montar. {cola}")

    @staticmethod
    def _last_json(stdout: str) -> dict:
        import json

        for linea in reversed((stdout or "").splitlines()):
            linea = linea.strip()
            if linea.startswith("{"):
                try:
                    data = json.loads(linea)
                except ValueError:
                    continue
                if isinstance(data, dict):
                    return data
        return {}


class VideoEnhanceWidget(TaskWidget):
    """Reescala, ajusta color, cambia los fotogramas y la velocidad de un vídeo."""

    #: El vídeo montado aquí (replay, rampa, rótulos…) listo para el Ensamblador.
    #: La ventana principal la conecta para precargarlo en esa pestaña.
    clips_ready = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._video: Path | None = None
        #: Última carpeta donde se ha dejado un resultado (``output/`` para los
        #: retoques y replays, ``projects/grabaciones/`` para comprimir/unir). La
        #: usa el botón «Abrir carpeta de resultados».
        self._last_dir: Path | None = None
        #: Sonda del vídeo elegido (duración, fps y si trae audio). La velocidad y
        #: el replay la necesitan: el audio hay que reajustarlo y el fragmento no
        #: puede salirse de la duración.
        self._info: dict = {}
        self._last_output: Path | None = None
        #: ¿Hay codificador por GPU? Se consulta una vez (``None`` = sin consultar).
        self._nvenc: bool | None = None
        self._build_ui()
        self._refresh_ai_availability()

    # ---- UI --------------------------------------------------------

    def _build_ui(self) -> None:
        # El formulario es muy alto (color, cámara lenta/repeticiones, rótulos,
        # comprimir y unir): sin scroll, la mitad de abajo quedaba inalcanzable.
        root = self._montar_contenido_scroll()

        fila = QHBoxLayout()
        self._btn_open = QPushButton("Elegir vídeo…")
        self._btn_open.clicked.connect(self._open)
        self._lbl_video = QLabel("(sin vídeo)")
        fila.addWidget(self._btn_open)
        fila.addWidget(self._lbl_video, 1)
        root.addLayout(fila)

        form = QFormLayout()
        self._spn_brightness = self._spin(vp.BRIGHTNESS_MIN, vp.BRIGHTNESS_MAX, vp.BRIGHTNESS_DEFAULT, 0.05)
        self._spn_contrast = self._spin(vp.CONTRAST_MIN, vp.CONTRAST_MAX, vp.CONTRAST_DEFAULT, 0.05)
        self._spn_saturation = self._spin(vp.SATURATION_MIN, vp.SATURATION_MAX, vp.SATURATION_DEFAULT, 0.05)
        self._spn_gamma = self._spin(vp.GAMMA_MIN, vp.GAMMA_MAX, vp.GAMMA_DEFAULT, 0.05)
        self._spn_sharpen = self._spin(vp.SHARPEN_MIN, vp.SHARPEN_MAX, vp.SHARPEN_DEFAULT, 0.1)
        self._spn_temperature = self._spin(
            vp.TEMPERATURE_MIN, vp.TEMPERATURE_MAX, vp.TEMPERATURE_DEFAULT, 5.0
        )
        self._spn_temperature.setToolTip(
            "Calidez de la imagen (temperatura de color): 0 es neutro; hacia +100 la\n"
            "imagen tira a cálida (naranja, luz de pabellón) y hacia −100 a fría (azul).\n"
            "Es el filtro colortemperature de ffmpeg."
        )
        form.addRow("Brillo:", self._spn_brightness)
        form.addRow("Contraste:", self._spn_contrast)
        form.addRow("Saturación:", self._spn_saturation)
        form.addRow("Gamma:", self._spn_gamma)
        self._spn_volume = QSpinBox()
        self._spn_volume.setRange(vp.VOLUME_MIN, vp.VOLUME_MAX)
        self._spn_volume.setValue(vp.VOLUME_DEFAULT)
        self._spn_volume.setSuffix(" %")
        self._spn_volume.setToolTip(
            "Volumen del audio del vídeo: 100 % = como está, 60 % = más bajo.\n"
            "Bájalo si se oye muy alto (o súbelo si casi no se oye). Al cambiarlo, el\n"
            "audio se recodifica (queda en AAC)."
        )
        form.addRow("Nitidez:", self._spn_sharpen)
        form.addRow("Calidez:", self._spn_temperature)
        form.addRow("Volumen:", self._spn_volume)

        self._cmb_denoise = QComboBox()
        for nivel in vp.DENOISE_LEVELS:
            self._cmb_denoise.addItem(nivel, nivel)
        form.addRow("Reduce-ruido:", self._cmb_denoise)

        self._cmb_fps = QComboBox()
        self._cmb_fps.addItem("Sin cambio", None)
        for fps in vp.FPS_CHOICES:
            self._cmb_fps.addItem(f"{fps} fps", fps)
        form.addRow("Fotogramas (FPS):", self._cmb_fps)

        self._spn_crf = QSpinBox()
        self._spn_crf.setRange(vp.CRF_MIN, vp.CRF_MAX)
        self._spn_crf.setValue(vp.CRF_DEFAULT)
        form.addRow("Calidad (CRF):", self._spn_crf)
        root.addLayout(form)
        root.addWidget(self._build_speed_group())
        root.addWidget(self._build_rolulo_group())
        root.addWidget(self._build_compress_group())
        root.addWidget(self._build_join_group())

        acciones = QHBoxLayout()
        self._btn_apply = QPushButton("Aplicar color/efectos (ffmpeg)")
        self._btn_apply.clicked.connect(self._apply_ffmpeg)
        acciones.addWidget(self._btn_apply)
        self._btn_send = QPushButton("Enviar al Ensamblador")
        self._btn_send.setEnabled(False)
        self._btn_send.setToolTip(
            "Precarga en el Ensamblador el último vídeo montado (replay, rampa,\n"
            "rótulos o cámara lenta) para añadirle música, voz o subtítulos."
        )
        self._btn_send.clicked.connect(self._send_to_assembler)
        acciones.addWidget(self._btn_send)
        acciones.addWidget(QLabel("Escalar IA:"))
        self._cmb_scale = QComboBox()
        for escala in vp.AI_SCALES:
            self._cmb_scale.addItem(f"x{escala}", escala)
        self._cmb_scale.setCurrentIndex(max(0, self._cmb_scale.findData(vp.AI_SCALE_DEFAULT)))
        acciones.addWidget(self._cmb_scale)
        self._btn_ai = QPushButton("Escalar con IA (Real-ESRGAN)")
        self._btn_ai.clicked.connect(self._apply_ai)
        acciones.addWidget(self._btn_ai)
        self._btn_cancel = QPushButton("Cancelar")
        self._btn_cancel.clicked.connect(self._cancel)
        acciones.addWidget(self._btn_cancel)
        # El resultado (retoque, replay, rótulos, compresión, unión) va a
        # ``output/`` o a ``projects/grabaciones/``. Sin este botón no había forma
        # de abrir la carpeta desde la pestaña y parecía que el fichero había
        # «desaparecido» al no verlo por ningún lado.
        self._btn_open_out = QPushButton("Abrir carpeta de resultados")
        self._btn_open_out.setEnabled(False)
        self._btn_open_out.clicked.connect(self._open_results_folder)
        acciones.addWidget(self._btn_open_out)
        acciones.addStretch(1)
        root.addLayout(acciones)

        # Barra de avance + tiempo (la monta la base, común a las tareas): antes
        # sólo había una etiqueta y no se sabía si iba por la mitad o atascado.
        self._montar_progreso(root)
        root.addStretch(1)
        self._btn_cancel.setEnabled(False)

    def _spin(self, lo, hi, value, step) -> QDoubleSpinBox:
        spn = QDoubleSpinBox()
        spn.setRange(lo, hi)
        spn.setSingleStep(step)
        spn.setValue(value)
        return spn

    def _build_speed_group(self) -> QGroupBox:
        """Velocidad (cámara lenta/rápida) y repeticiones de un momento."""
        grupo = QGroupBox("Cámara lenta y repeticiones")
        grid = QFormLayout(grupo)

        self._spn_speed = QDoubleSpinBox()
        self._spn_speed.setRange(vp.SPEED_MIN, vp.SPEED_MAX)
        self._spn_speed.setSingleStep(vp.SPEED_STEP)
        self._spn_speed.setDecimals(2)
        self._spn_speed.setValue(vp.SPEED_DEFAULT)
        self._spn_speed.setSuffix(" x")
        self._spn_speed.setToolTip(
            "Multiplicador de reproducción: 0.5x es la mitad de rápido (cámara\n"
            "lenta) y 2x el doble. El audio se reajusta con atempo, así que no se\n"
            "desincroniza del vídeo."
        )
        grid.addRow("Velocidad:", self._spn_speed)

        self._spn_zoom = self._spin(vp.ZOOM_MIN, vp.ZOOM_MAX, vp.ZOOM_DEFAULT, vp.ZOOM_STEP)
        self._spn_zoom.setSuffix(" x")
        self._spn_zoom.setToolTip(
            "Acerca la imagen recortando el centro (1x = encuadre original). Con el\n"
            "zoom puesto, la repetición se ve más de cerca."
        )
        grid.addRow("Zoom:", self._spn_zoom)

        self._chk_smooth = QCheckBox("Suavizar el movimiento (interpolar)")
        self._chk_smooth.setChecked(True)
        self._chk_smooth.setToolTip(
            "Inventa los fotogramas que faltan (minterpolate) antes de estirar el\n"
            "tiempo, así la cámara lenta se ve continua y no a saltos. Tarda\n"
            "bastante más y sólo aplica a la velocidad lenta."
        )
        grid.addRow(self._chk_smooth)

        self._btn_speed = QPushButton("Aplicar velocidad")
        self._btn_speed.clicked.connect(self._apply_speed)
        grid.addRow(self._btn_speed)

        self._spn_start = self._spin(0.0, 36000.0, 0.0, 0.5)
        self._spn_start.setSuffix(" s")
        self._spn_end = self._spin(0.0, 36000.0, vp.REPLAY_CLIP_DEFAULT_S, 0.5)
        self._spn_end.setSuffix(" s")
        self._spn_times = QSpinBox()
        self._spn_times.setRange(vp.REPLAY_MIN_TIMES, vp.REPLAY_MAX_TIMES)
        self._spn_times.setValue(vp.REPLAY_TIMES_DEFAULT)

        fragmento = QHBoxLayout()
        fragmento.addWidget(QLabel("Del"))
        fragmento.addWidget(self._spn_start)
        fragmento.addWidget(QLabel("al"))
        fragmento.addWidget(self._spn_end)
        fragmento.addWidget(QLabel("· Repeticiones:"))
        fragmento.addWidget(self._spn_times)
        fragmento.addStretch(1)
        grid.addRow("Fragmento:", fragmento)

        self._chk_slow_last = QCheckBox("Última repetición en cámara lenta")
        self._chk_slow_last.setChecked(True)
        self._chk_slow_last.setToolTip(
            f"La última vuelta va a {vp.REPLAY_SLOW_SPEED:g}x: primero se ve la\n"
            "jugada entera y después el momento clave, más despacio."
        )
        grid.addRow(self._chk_slow_last)

        self._btn_replay = QPushButton("Crear repetición")
        self._btn_replay.clicked.connect(self._apply_replay)
        grid.addRow(self._btn_replay)

        self._btn_ramp = QPushButton("Crear rampa (sólo el tramo lento)")
        self._btn_ramp.setToolTip(
            "Deja el vídeo a velocidad normal y ralentiza únicamente el fragmento:\n"
            "es la rampa de las retransmisiones (llega la jugada y baja la velocidad)."
        )
        self._btn_ramp.clicked.connect(self._apply_ramp)
        grid.addRow(self._btn_ramp)
        return grupo

    def _build_rolulo_group(self) -> QGroupBox:
        """Rótulos de texto sobre el vídeo (marcador, minuto, nombres)."""
        grupo = QGroupBox("Rótulos (marcador, minuto, nombres)")
        grid = QFormLayout(grupo)

        self._txt_rolulos = QPlainTextEdit()
        self._txt_rolulos.setPlainText(vp.ROLULO_EXAMPLE)
        self._txt_rolulos.setFixedHeight(78)
        self._txt_rolulos.setToolTip(
            "Una línea por rótulo: «minuto:segundo frase» (las líneas que empiezan\n"
            "por # se ignoran). Ejemplo:\n" + vp.ROLULO_EXAMPLE
        )
        grid.addRow("Rótulos:", self._txt_rolulos)

        self._spn_rolulo_seconds = self._spin(0.5, 10.0, vp.ROLULO_DEFAULT_SECONDS, 0.5)
        self._spn_rolulo_seconds.setSuffix(" s")
        self._spn_rolulo_size = QSpinBox()
        self._spn_rolulo_size.setRange(12, 96)
        self._spn_rolulo_size.setValue(vp.ROLULO_FONTSIZE)
        self._cmb_rolulo_pos = QComboBox()
        for clave, etiqueta in asp.POSITIONS:
            self._cmb_rolulo_pos.addItem(etiqueta, clave)
        self._cmb_rolulo_color = QComboBox()
        for clave, etiqueta in asp.COLORS:
            self._cmb_rolulo_color.addItem(etiqueta, clave)

        estilo = QHBoxLayout()
        estilo.addWidget(QLabel("Duración:"))
        estilo.addWidget(self._spn_rolulo_seconds)
        estilo.addWidget(QLabel("Posición:"))
        estilo.addWidget(self._cmb_rolulo_pos)
        estilo.addWidget(QLabel("Tamaño:"))
        estilo.addWidget(self._spn_rolulo_size)
        estilo.addWidget(QLabel("Color:"))
        estilo.addWidget(self._cmb_rolulo_color)
        estilo.addStretch(1)
        grid.addRow("", estilo)

        self._btn_rolulos = QPushButton("Quemar rótulos")
        self._btn_rolulos.clicked.connect(self._apply_rolulos)
        grid.addRow(self._btn_rolulos)
        return grupo

    def _build_compress_group(self) -> QGroupBox:
        """Recodificar el vídeo para que ocupe menos (sin tocar la imagen)."""
        grupo = QGroupBox("Comprimir (recodificar más pequeño)")
        grid = QFormLayout(grupo)

        self._cmb_quality = QComboBox()
        for clave, etiqueta, crf, cq in vp.COMPRESSION_QUALITIES:
            self._cmb_quality.addItem(f"{etiqueta} (CRF {crf} / CQ {cq})", clave)
        self._cmb_quality.setCurrentIndex(
            max(0, self._cmb_quality.findData(vp.COMPRESSION_DEFAULT))
        )
        self._cmb_quality.setToolTip(
            "Más valor = archivo más pequeño y algo menos de calidad. Medido con un\n"
            "trozo de cámara a 1080p50 y 25 Mbps: «Equilibrada» (CRF 23 / CQ 29) deja\n"
            "el archivo en un tercio (108 MB -> 35 MB) y «Alta» en la mitad."
        )
        grid.addRow("Calidad:", self._cmb_quality)

        self._cmb_encoder = QComboBox()
        self._cmb_encoder.addItem("CPU (libx264, mejor calidad por MB)", vp.ENCODER_CPU)
        if self._nvenc_available():
            # Medido en esta máquina (RTX 5080): 34 s de 1080p50 en 4,8 s (GPU)
            # frente a 6,3-7,4 s (CPU), con el CQ ya compensado para igualar tamaño.
            self._cmb_encoder.addItem("GPU (NVENC, más rápido)", vp.ENCODER_GPU)
        self._cmb_encoder.setToolTip(
            "La GPU va entre un 20 % y un 50 % más rápida en este equipo y deja un\n"
            "archivo de tamaño parecido (el CQ ya va compensado); la CPU da algo más\n"
            "de calidad por MB, así que para la entrega final puede compensar."
        )
        grid.addRow("Codificador:", self._cmb_encoder)

        self._btn_compress = QPushButton("Comprimir vídeo")
        self._btn_compress.setToolTip(
            "Recodifica el vídeo completo sin cambiarle nada: solo baja el bitrate.\n"
            "El resultado va a projects/grabaciones/ (no a output/, que se limpia\n"
            "por tamaño y antigüedad). El audio se pasa a AAC a 192 k."
        )
        self._btn_compress.clicked.connect(self._apply_compress)

        self._btn_compress_many = QPushButton("Comprimir varios (lote)…")
        self._btn_compress_many.setToolTip(
            "Elige varios ficheros (los trozos de la cámara) y los comprime seguidos,\n"
            "de uno en uno y con el mismo ajuste, sin tener que ir llamando al botón\n"
            "diez veces. Se puede cancelar entre trozos."
        )
        self._btn_compress_many.clicked.connect(self._apply_compress_many)

        fila_botones = QHBoxLayout()
        fila_botones.addWidget(self._btn_compress)
        fila_botones.addWidget(self._btn_compress_many)
        fila_botones.addStretch(1)
        grid.addRow("", fila_botones)
        return grupo

    def _nvenc_available(self) -> bool:
        """¿El ffmpeg instalado trae H.264 por GPU? Se consulta una sola vez."""
        if self._nvenc is None:
            self._nvenc = False
            try:
                proc = subprocess.run(
                    [_ffmpeg_exe(), "-hide_banner", "-encoders"],
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=30, **no_window_kwargs(),
                )
                self._nvenc = vp.has_nvenc(proc.stdout or "")
            except (OSError, subprocess.SubprocessError) as exc:
                logger.debug("No se pudo consultar los codificadores: %s", exc)
        return bool(self._nvenc)

    def _build_join_group(self) -> QGroupBox:
        """Unir grabaciones troceadas (los ``.MTS`` de una cámara) en un MP4."""
        grupo = QGroupBox("Unir grabaciones troceadas (.MTS y similares)")
        caja = QVBoxLayout(grupo)

        self._lista_union = QListWidget()
        self._lista_union.setFixedHeight(84)
        self._lista_union.setToolTip(
            "Los trozos se unen en orden alfabético (00000, 00001…), que es como los\n"
            "numera la cámara. El vídeo se copia tal cual: no se pierde calidad, y el\n"
            "audio AC-3 de cámara se convierte a AAC para que lo abra todo."
        )
        caja.addWidget(self._lista_union)

        self._chk_join_reencode = QCheckBox("Igualar los trozos (recomprimir)")
        self._chk_join_reencode.setToolTip(
            "Márcalo si los trozos vienen ya retocados: una rampa, una repetición o\n"
            "una cámara lenta cambian el ritmo y un escalado cambia el tamaño, así que\n"
            "la copia directa falla o da saltos. Si lo dejas sin marcar, la app intenta\n"
            "copiar primero (mucho más rápido) y sólo recomprime si hace falta."
        )
        caja.addWidget(self._chk_join_reencode)

        fila = QHBoxLayout()
        self._btn_add_clips = QPushButton("Añadir grabaciones…")
        self._btn_add_clips.clicked.connect(self._add_clips)
        self._btn_remove_clip = QPushButton("Quitar seleccionada")
        self._btn_remove_clip.clicked.connect(self._remove_clip)
        self._btn_join = QPushButton("Unir en un solo MP4")
        self._btn_join.clicked.connect(self._join_clips)
        fila.addWidget(self._btn_add_clips)
        fila.addWidget(self._btn_remove_clip)
        fila.addWidget(self._btn_join)
        fila.addStretch(1)
        caja.addLayout(fila)
        return grupo

    def _busy_widgets(self) -> tuple:
        return (
            self._btn_open, self._btn_apply, self._btn_ai, self._cmb_scale,
            self._cmb_denoise, self._cmb_fps, self._spn_crf,
            self._spn_brightness, self._spn_contrast, self._spn_saturation,
            self._spn_gamma, self._spn_sharpen, self._spn_temperature,
            self._spn_volume,
            self._btn_speed, self._spn_speed, self._chk_smooth, self._spn_zoom,
            self._btn_replay, self._spn_start, self._spn_end, self._spn_times,
            self._chk_slow_last, self._btn_ramp,
            self._btn_rolulos, self._txt_rolulos, self._spn_rolulo_seconds,
            self._cmb_rolulo_pos, self._spn_rolulo_size, self._cmb_rolulo_color,
            self._btn_add_clips, self._btn_remove_clip, self._btn_join,
            self._lista_union, self._chk_join_reencode,
            self._btn_compress, self._btn_compress_many,
            self._cmb_quality, self._cmb_encoder,
            self._btn_send,
        )

    def _busy_only_widgets(self) -> tuple:
        return (self._btn_cancel,)

    def _on_busy_changed(self, busy: bool) -> None:
        """Al terminar, vuelve a apagar lo que no esté disponible.

        ``_set_busy(False)`` reactiva todos los controles, así que sin esto el
        botón de IA quedaba habilitado tras el primer trabajo aunque faltase
        ``venv-photo`` (y al pulsarlo sólo salía «no disponible»), y el de enviar al
        Ensamblador aparecía activo aunque todavía no hubiera nada montado.
        """
        if not busy:
            self._refresh_ai_availability()
            self._btn_send.setEnabled(self._last_output is not None)

    # ---- Disponibilidad IA -----------------------------------------

    def _ai_paths(self) -> tuple[Path, Path] | None:
        python = PROJECT_ROOT / VIDEO_PYTHON
        worker = PROJECT_ROOT / VIDEO_WORKER
        if python.is_file() and worker.is_file():
            return python, worker
        return None

    def _refresh_ai_availability(self) -> None:
        if self._ai_paths() is None:
            self._btn_ai.setEnabled(False)
            self._btn_ai.setToolTip(
                "Escalado IA no disponible: falta venv-photo (realesrgan) o "
                "tools/video_upscale_worker.py."
            )

    # ---- Acciones --------------------------------------------------

    def _open(self) -> None:
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Elegir vídeo", "", vu.VIDEO_FILTER_ALL
        )
        if ruta:
            self._video = Path(ruta)
            self._lbl_video.setText(self._video.name)
            self._probe_video()

    def _probe_video(self) -> None:
        """Lee duración, fps y audio del vídeo (los usan la velocidad y el replay)."""
        self._info = {}
        if self._video is None:
            return
        try:
            self._info = VideoAssembler().probe(self._video)
        except EngineError as exc:
            logger.debug("No se pudo sondear el vídeo: %s", exc)
            return
        duracion = self._info.get("duration") or 0.0
        if duracion:
            # El fragmento del replay no puede salirse del vídeo: se acotan los
            # controles a lo que dura (y se propone un fragmento corto al final).
            self._spn_start.setMaximum(duracion)
            self._spn_end.setMaximum(duracion)
            self._spn_end.setValue(min(vp.REPLAY_CLIP_DEFAULT_S, duracion))

    def _has_audio(self) -> bool:
        """¿El vídeo trae sonido? Sin sonda se asume que sí (es lo habitual)."""
        return bool(self._info.get("has_audio", True))

    def _zoom_size(self) -> tuple[int, int] | None:
        """Tamaño del vídeo, para que el zoom devuelva **la misma** resolución.

        Sin esto el reescalado redondea y la salida baila un par de píxeles
        (medido: 1,5x sobre 640x360 daba 638x360), lo que rompería un ``concat``
        posterior en el Ensamblador, que exige que todos los clips coincidan.
        """
        ancho, alto = self._info.get("width"), self._info.get("height")
        if ancho and alto:
            return (int(ancho), int(alto))
        return None

    def _require_video(self) -> Path | None:
        if self._video is None:
            self._set_status("Elige un vídeo primero.")
            return None
        return self._video

    def _out_path(self, sufijo: str) -> Path:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        marca = datetime.now().strftime("%Y%m%d_%H%M%S")
        return OUTPUT / f"{self._video.stem}_{sufijo}_{marca}.mp4"

    def _apply_ffmpeg(self) -> None:
        video = self._require_video()
        if video is None:
            return
        filtros = vp.build_filters(
            adjust=vp.ColorAdjust(
                brightness=self._spn_brightness.value(),
                contrast=self._spn_contrast.value(),
                saturation=self._spn_saturation.value(),
                gamma=self._spn_gamma.value(),
                temperature=self._spn_temperature.value(),
            ),
            denoise=self._cmb_denoise.currentData(),
            sharpen=self._spn_sharpen.value(),
            fps=self._cmb_fps.currentData(),
        )
        # El volumen es un filtro de audio: puede ir solo, sin tocar la imagen.
        volumen = vp.volume_filter(self._spn_volume.value())
        if not filtros and not volumen:
            self._set_status("No hay ajustes que aplicar.")
            return
        self._start(FfmpegEnhanceWorker(
            video, self._out_path("retoque"), filtros, self._spn_crf.value(),
            audio_filter=volumen,
        ))

    def _apply_speed(self) -> None:
        """Aplica la velocidad elegida (cámara lenta/rápida) al vídeo completo."""
        video = self._require_video()
        if video is None:
            return
        velocidad = self._spn_speed.value()
        filtros = vp.build_filters(
            speed=velocidad,
            smooth=self._chk_smooth.isChecked(),
            source_fps=self._info.get("fps"),
            zoom=self._spn_zoom.value(),
        )
        if not filtros:
            self._set_status("La velocidad es 1x: cambia el valor para aplicar algo.")
            return
        audio = vp.audio_speed_filter(velocidad)
        if audio is not None and not self._has_audio():
            # Pedir un filtro de audio en un vídeo mudo hace fallar ffmpeg: se
            # cambia la velocidad de la imagen y se deja el fichero sin sonido.
            audio = None
        sufijo = "lenta" if velocidad < vp.SPEED_DEFAULT else "rapida"
        self._start(FfmpegEnhanceWorker(
            video, self._out_path(sufijo), filtros, self._spn_crf.value(),
            audio_filter=audio,
        ))

    def _apply_replay(self) -> None:
        """Repite el fragmento elegido, con la última vuelta en cámara lenta."""
        video = self._require_video()
        if video is None:
            return
        try:
            grafo = vp.replay_filter_complex(
                self._spn_start.value(),
                self._spn_end.value(),
                self._spn_times.value(),
                slow_last=self._chk_slow_last.isChecked(),
                has_audio=self._has_audio(),
                smooth=self._chk_smooth.isChecked(),
                source_fps=self._info.get("fps"),
                zoom=self._spn_zoom.value(),
                zoom_size=self._zoom_size(),
            )
        except ValueError as exc:
            self._set_status(str(exc))
            return
        self._start(ReplayWorker(
            video, self._out_path("replay"), grafo, self._spn_crf.value()
        ))

    def _apply_ramp(self) -> None:
        """Ralentiza sólo el tramo elegido y deja el resto a velocidad normal."""
        video = self._require_video()
        if video is None:
            return
        duracion = self._info.get("duration")
        if not duracion:
            self._set_status("No se pudo leer la duración del vídeo.")
            return
        try:
            grafo = vp.speed_ramp_filter_complex(
                duracion,
                self._spn_start.value(),
                self._spn_end.value(),
                has_audio=self._has_audio(),
                smooth=self._chk_smooth.isChecked(),
                source_fps=self._info.get("fps"),
                zoom=self._spn_zoom.value(),
                zoom_size=self._zoom_size(),
            )
        except ValueError as exc:
            self._set_status(str(exc))
            return
        self._start(ReplayWorker(
            video, self._out_path("rampa"), grafo, self._spn_crf.value()
        ))

    def _apply_rolulos(self) -> None:
        """Quema los rótulos escritos (una línea por rótulo)."""
        video = self._require_video()
        if video is None:
            return
        segundos = self._spn_rolulo_seconds.value()
        try:
            rotulos = vp.parse_rolulos(
                self._txt_rolulos.toPlainText(),
                default_seconds=segundos,
                duration=self._info.get("duration"),
            )
        except ValueError as exc:
            self._set_status(str(exc))
            return
        estilo = asp.build_subtitle_style(
            self._spn_rolulo_size.value(),
            self._cmb_rolulo_pos.currentData(),
            self._cmb_rolulo_color.currentData(),
        )
        self._start(RoluloWorker(
            video, self._out_path("rotulos"),
            vp.rolulos_srt(rotulos, default_seconds=segundos),
            estilo, self._spn_crf.value(),
        ))

    def _apply_compress_many(self) -> None:
        """Comprime varios ficheros seguidos (lote), sin tocar la imagen de ninguno."""
        rutas, _ = QFileDialog.getOpenFileNames(
            self, "Comprimir varios", "", vu.VIDEO_FILTER_ALL
        )
        if not rutas:
            return
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        marca = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Orden por nombre: 00000, 00001…, que es el orden de grabación de la cámara.
        trabajos: list[tuple[Path, Path]] = []
        usados: set[Path] = set()
        for origen in vp.join_paths_ordered(rutas):
            # El destino se construye con el nombre del origen, así que dos ficheros
            # con el mismo nombre en carpetas distintas (o repetir el lote dentro
            # del mismo segundo) colisionarían y uno pisaría al otro: se desambigua
            # con un sufijo. También se evita escribir **sobre** el propio origen.
            destino = RECORDINGS_DIR / f"{origen.stem}_comprimido_{marca}.mp4"
            extra = 2
            while destino in usados or destino.resolve() == origen.resolve():
                destino = (
                    RECORDINGS_DIR / f"{origen.stem}_comprimido_{marca}_{extra}.mp4"
                )
                extra += 1
            usados.add(destino)
            trabajos.append((origen, destino))
        codificador = self._cmb_encoder.currentData()
        calidad = vp.quality_value(self._cmb_quality.currentData(), codificador)
        self._start(
            BatchCompressWorker(trabajos, calidad, codificador),
            on_result=self._on_batch_done,
        )

    def _on_batch_done(self, resumen) -> None:
        """El lote no es un vídeo suelto: se informa y no se toca «el último».

        Se recuerda la carpeta de destino y se ofrece abrirla: los comprimidos van
        a ``projects/grabaciones/`` (fuera de ``output/``), y no verlos en la
        carpeta que uno miraba hacía pensar que habían «desaparecido».
        """
        self._last_dir = RECORDINGS_DIR
        self._btn_open_out.setEnabled(True)
        carpeta = RECORDINGS_DIR.name
        self._set_status(f"{resumen} · en projects/{carpeta}/")

    def _apply_compress(self) -> None:
        """Recodifica el vídeo para que pese menos (sin filtros: solo calidad)."""
        video = self._require_video()
        if video is None:
            return
        codificador = self._cmb_encoder.currentData()
        calidad = vp.quality_value(self._cmb_quality.currentData(), codificador)
        # Va a projects/grabaciones/: es un máster de trabajo, y output/ se limpia
        # por antigüedad y tamaño (la compresión de un partido no debe desaparecer).
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        marca = datetime.now().strftime("%Y%m%d_%H%M%S")
        destino = RECORDINGS_DIR / f"{video.stem}_comprimido_{marca}.mp4"
        self._start(FfmpegEnhanceWorker(
            video, destino, "", calidad,
            encoder=self._cmb_encoder.currentData(), audio_codec="aac",
        ))

    def _add_clips(self) -> None:
        """Añade grabaciones a la lista de unión (se ordenan por nombre)."""
        rutas, _ = QFileDialog.getOpenFileNames(
            self, "Añadir grabaciones", "", vu.VIDEO_FILTER_ALL
        )
        for ruta in vp.join_paths_ordered(rutas):
            texto = str(ruta)
            if not self._lista_union.findItems(texto, Qt.MatchFlag.MatchExactly):
                self._lista_union.addItem(texto)
        # La cámara numera los trozos 00000, 00001…: ordenar la lista es unir en el
        # orden correcto (el selector puede devolverlos en cualquier otro).
        self._lista_union.sortItems()
        if rutas:
            self._set_status(f"{self._lista_union.count()} grabaciones en la lista.")

    def _remove_clip(self) -> None:
        for item in self._lista_union.selectedItems():
            self._lista_union.takeItem(self._lista_union.row(item))

    def _join_clips(self) -> None:
        """Une las grabaciones de la lista en un solo MP4 (con copia de vídeo)."""
        rutas = [
            self._lista_union.item(i).text() for i in range(self._lista_union.count())
        ]
        if len(rutas) < 2:
            self._set_status("Añade al menos dos grabaciones para unirlas.")
            return
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        marca = datetime.now().strftime("%Y%m%d_%H%M%S")
        destino = RECORDINGS_DIR / f"grabacion_unida_{marca}.mp4"
        self._start(JoinWorker(
            rutas, destino, reencode=self._chk_join_reencode.isChecked()
        ))

    def _apply_ai(self) -> None:
        video = self._require_video()
        if video is None:
            return
        paths = self._ai_paths()
        if paths is None:
            self._set_status("Escalado IA no disponible (falta venv-photo).")
            return
        python, worker = paths
        self._start(AiUpscaleWorker(
            video, self._out_path("ia"), int(self._cmb_scale.currentData()),
            python, worker, PROJECT_ROOT / VIDEO_MODELS_DIR,
        ))

    def _start(self, worker: ThreadWorker, on_result=None) -> None:
        self._worker = worker
        worker.progress.connect(self._on_progress)
        # El resultado por defecto es un vídeo (habilita «Enviar al Ensamblador»);
        # un lote devuelve un resumen de texto y trae su propio manejador.
        worker.result.connect(on_result or self._on_result)
        worker.failed.connect(self._on_error)
        worker.finished.connect(self._on_finished)
        # La barra arranca en indeterminado: el worker dirá el porcentaje real en
        # cuanto ffmpeg empiece a informar. El reloj, a contar desde cero.
        self._empezar_progreso()
        self._set_busy(True)
        self._set_status("Trabajando…")
        worker.start()

    def _on_result(self, path) -> None:
        """Recuerda el vídeo montado y deja enviarlo al Ensamblador."""
        self._last_output = Path(path) if path else None
        self._btn_send.setEnabled(self._last_output is not None)
        if self._last_output is not None:
            self._last_dir = self._last_output.parent
            self._btn_open_out.setEnabled(True)
        self._set_status(f"Listo: {path}")

    def _open_results_folder(self) -> None:
        """Abre en el Explorador la carpeta donde se ha dejado el resultado."""
        destino = self._last_dir or OUTPUT
        if not destino.is_dir():
            self._set_status(f"La carpeta todavía no existe: {destino}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(destino)))
        self._set_status(f"Carpeta abierta: {destino}")

    def _send_to_assembler(self) -> None:
        if self._last_output is None:
            self._set_status("Todavía no hay ningún vídeo montado.")
            return
        self.clips_ready.emit(str(self._last_output))
        self._set_status("Vídeo enviado al Ensamblador.")

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.requestInterruption()
