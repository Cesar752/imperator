"""Pestaña **Cuaderno**: análisis cinematográfico local («Lapian Notes»).

Flujo (todo en el equipo, sin subir nada a ningún servidor):

1. Se elige un MP4.
2. **Generar paquete**: extrae un fotograma por segundo (ffmpeg) y monta un ZIP
   con los fotogramas, las instrucciones y el esquema JSON para enviárselo a una
   IA externa si se quiere.
3. **Analizar en local**: sin salir de Imperator, describe los fotogramas con
   Ollama Vision y compone el JSON del análisis con Ollama (chat).
4. **Importar análisis**: carga el JSON devuelto por cualquier IA.
5. Se muestran líneas narrativas, árbol de estructura, curva de emoción y
   segmentos/escenas, y se puede **exportar a Markdown**.

El *modelo* y el *montaje* viven en :mod:`services.notes_params`; aquí solo se
ejecuta (ffmpeg y motores) y se pinta el resultado.
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QColor, QDesktopServices, QPainter, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.frame_cache import FrameCache
from core.frame_extractor import FrameExtractor
from core.paths import PROJECT_ROOT
from engines.base_engine import EngineError
from services import chat_engine as ce
from services import notes_params as nts
from services import preferences as prefs
from services import scoreboard as sb
from services import sport_params as sp
from services import video_understanding as vu
from services.video_understanding import VIDEO_FILTER_ALL, clip_labels_text, srt_to_text
from ui import discipline
from ui.task_widget import TaskWidget
from ui.theme import semantic
from ui.workers import ThreadWorker

logger = logging.getLogger(__name__)


def _agotado(exc: Exception) -> bool:
    """¿El fallo es «se agotó el espacio/ventana» (o un JSON cortado)?

    Solo entonces se reintenta con un prompt más corto: para otros errores
    (conexión, modelo inexistente...) reintentar no ayudaría.
    """
    if isinstance(exc, nts.NotesError):
        return True  # respuesta vacía o JSON cortado: no había nada que leer
    texto = str(exc).lower()
    return "agot" in texto or "ventana" in texto or "context" in texto


def componer_cuaderno(chat, prompts, *, parsear, aviso=None):
    """Pide el cuaderno al chat probando ``prompts`` en orden (de más a menos largo).

    ``chat(prompt) -> str`` es la llamada al motor y ``parsear(respuesta)`` el
    parseo. Si el motor se queda sin espacio (o la respuesta sale cortada), se
    prueba con el prompt siguiente (más corto). Lanza el último error si ninguno vale.
    """
    ultimo: Exception | None = None
    for indice, prompt in enumerate(prompts):
        try:
            return parsear(chat(prompt))
        except Exception as exc:  # noqa: BLE001 - se re-lanza lo que no sea agotamiento
            if not _agotado(exc):
                raise
            ultimo = exc
            if aviso is not None and indice + 1 < len(prompts):
                aviso(indice + 1)
    if ultimo is None:
        raise EngineError("No hay prompt con el que componer el cuaderno.")
    raise ultimo

#: Carpeta de los cuadernos y paquetes (persistente, fuera de output/).
NOTES_DIR = PROJECT_ROOT / "projects" / "notes"

#: Carpeta donde Comprensión deja su percepción estructurada (comprension.json),
#: que el Cuaderno puede reutilizar para no repetir YOLO/CLIP/OCR/Whisper.
UNDERSTANDING_DIR = PROJECT_ROOT / "projects" / "comprension"

#: Fichero del análisis dentro de la carpeta del cuaderno.
ANALYSIS_FILENAME = "analisis.json"

#: Filtro del diálogo «Elegir vídeo». Se mantiene alineado con las demás
#: pestañas que leen vídeo (``ui.understanding_widget``, ``ui.video_enhance_widget``):
#: el cuaderno también analiza AVI (y MKV/MOV/M4V/WebM), no solo MP4.
VIDEO_FILTER = "Vídeo (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;Todos (*)"


class EmotionCurveWidget(QWidget):
    """Dibuja la curva de emoción del espectador (t → intensidad)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._points: list[nts.EmotionPoint] = []
        self.setMinimumHeight(180)

    def set_points(self, points) -> None:
        self._points = list(points or [])
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - override de Qt
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(40, 16, -16, -28)
        painter.fillRect(self.rect(), QColor("#1e1e1e"))
        painter.setPen(QPen(QColor("#555"), 1))
        painter.drawRect(rect)

        if not self._points:
            painter.setPen(QColor("#aaa"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Sin datos de emoción")
            painter.end()
            return

        t_max = max((p.t_s for p in self._points), default=1.0) or 1.0
        rango = nts.EMOTION_MAX - nts.EMOTION_MIN or 1.0

        def punto(p: nts.EmotionPoint):
            x = rect.left() + (p.t_s / t_max) * rect.width()
            frac = (p.intensidad - nts.EMOTION_MIN) / rango
            y = rect.bottom() - frac * rect.height()
            return int(x), int(y)

        painter.setPen(QPen(QColor("#4e79a7"), 2))
        prev = None
        for p in self._points:
            actual = punto(p)
            if prev is not None:
                painter.drawLine(prev[0], prev[1], actual[0], actual[1])
            prev = actual

        painter.setBrush(QColor("#f28e2b"))
        painter.setPen(QPen(QColor("#f28e2b"), 1))
        for p in self._points:
            x, y = punto(p)
            painter.drawEllipse(x - 3, y - 3, 6, 6)
            if p.etiqueta:
                painter.setPen(QColor("#ddd"))
                painter.drawText(x + 4, y - 4, p.etiqueta)
                painter.setPen(QPen(QColor("#f28e2b"), 1))

        painter.setPen(QColor("#aaa"))
        painter.drawText(rect.left(), rect.bottom() + 18, "0")
        painter.drawText(rect.right() - 40, rect.bottom() + 18, nts.format_timestamp(t_max))
        painter.end()


class PackageWorker(ThreadWorker):
    """Extrae los fotogramas (1 fps) y compone el ZIP del cuaderno."""

    def __init__(
        self,
        video: Path,
        *,
        extractor: FrameExtractor,
        fps: float,
        max_frames: int,
        include_audio: bool,
        whisper_engine,
        out_dir: Path,
        sport: str = "",
        context: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._video = video
        self._extractor = extractor
        self._fps = fps
        self._max_frames = max_frames
        self._include_audio = include_audio
        self._whisper = whisper_engine
        self._out_dir = out_dir
        #: Disciplina (para incluir sus reglas en el paquete) y datos del usuario.
        self._sport = sport
        self._context = context

    def work(self) -> dict:
        duracion = self._extractor.duration(self._video)
        if not duracion or duracion <= 0:
            raise EngineError("No se pudo leer la duración del vídeo (¿está dañado?).")
        instantes = nts.sample_frames(duracion, self._fps, self._max_frames)
        transcript = self._transcribe(duracion)

        marcados: list[tuple[float, Path]] = []
        total = len(instantes)
        for i, t in enumerate(instantes):
            self._check_cancel()
            # Caché: re-extraer el mismo fotograma(vídeo, t) no vuelve a decodificar.
            marcados.append((t, self._extractor.cached_frame(self._video, t)))
            self.progress.emit(int((i + 1) / max(1, total) * 100), "Extrayendo fotogramas…")

        zip_path = self._out_dir / f"{self._video.stem}_paquete.zip"
        nts.build_package(
            zip_path,
            frames=marcados,
            video_name=self._video.name,
            duration_s=duracion,
            transcript=transcript,
            origin="imperator:cuaderno",
            sport=self._sport,
            context=self._context,
        )
        return {"zip": zip_path, "frames": len(marcados), "transcript": bool(transcript)}

    def _transcribe(self, duracion: float) -> str:
        if not self._include_audio or self._whisper is None:
            return ""
        if not self._whisper.is_available():
            logger.info("Cuaderno: Whisper no está disponible; paquete sin transcripción.")
            return ""
        with tempfile.TemporaryDirectory(prefix="imperator_notes_audio_") as tmp:
            wav = Path(tmp) / "audio.wav"
            self._check_cancel()
            self.progress.emit(0, "Extrayendo audio…")
            self._extractor.extract_audio(self._video, wav)
            srt = self._whisper.transcribe(wav)
            try:
                return srt_to_text(Path(srt).read_text(encoding="utf-8", errors="replace"))
            except OSError:
                return ""


class AnalyzeWorker(ThreadWorker):
    """Analiza el vídeo en local: visión por fotograma + chat que compone el JSON."""

    def __init__(
        self,
        video: Path,
        *,
        extractor: FrameExtractor,
        vision_engine,
        chat_engine,
        whisper_engine,
        vision_frames: int,
        include_audio: bool,
        model: str | None,
        out_dir: Path,
        sport: str = "",
        context: str = "",
        instructions: str = "",
        objects_engine=None,
        labels_engine=None,
        ocr_engine=None,
        clip_labels: str = "",
        perception=None,
        perception_transcript: str = "",
        perception_marcador=None,
        perception_juego_inicio=None,
        perception_descanso=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._video = video
        self._extractor = extractor
        self._vision = vision_engine
        self._chat = chat_engine
        self._whisper = whisper_engine
        self._vision_frames = vision_frames
        self._include_audio = include_audio
        self._model = model
        self._out_dir = out_dir
        #: Disciplina (reglas) y contexto del usuario, para los prompts de visión
        #: y de composición del cuaderno.
        self._sport = sport
        self._context = context
        #: Instrucciones del usuario (qué contar y con qué prioridad): van al prompt
        #: de cada fotograma y al que compone el cuaderno.
        self._instructions = str(instructions or "")
        #: Motores de percepción opcionales (objetos/etiquetas/OCR): añaden datos
        #: estructurados a cada fotograma, además de la descripción de la visión.
        self._objects_engine = objects_engine
        self._labels_engine = labels_engine
        self._ocr_engine = ocr_engine
        self._clip_labels = str(clip_labels or "")
        #: Percepción ``(t_s, texto)`` reutilizada de Comprensión (o vacía): si
        #: viene, no se repiten YOLO/CLIP/OCR. ``perception_transcript`` permite
        #: además reutilizar la transcripción de Whisper.
        self._perception = list(perception or [])
        self._perception_transcript = str(perception_transcript or "")
        #: Serie del marcador reutilizada de Comprensión (lista de dicts
        #: ``{t_s, izquierda, derecha}``): evita re-leer el marcador por OCR.
        self._perception_marcador = list(perception_marcador or [])
        self._perception_juego_inicio = perception_juego_inicio
        self._perception_descanso = perception_descanso

    def work(self) -> dict:
        if self._chat is None or not self._chat.is_available():
            raise EngineError("No hay motor de chat (Ollama) disponible para el cuaderno.")
        if self._vision is None or not self._vision.is_available():
            raise EngineError("No hay motor de visión (Ollama Vision) disponible.")

        duracion = self._extractor.duration(self._video)
        if not duracion or duracion <= 0:
            raise EngineError("No se pudo leer la duración del vídeo (¿está dañado?).")

        instantes = nts.spread(
            nts.sample_frames(duracion, nts.DEFAULT_FPS, nts.DEFAULT_MAX_FRAMES),
            self._vision_frames,
        )
        notas: list[tuple[float, str]] = []
        #: Percepción por fotograma (incluye el texto del OCR): de aquí sale el
        #: marcador real, en vez de fiarnos de lo que «crea» el modelo de visión.
        lecturas_vision: list[tuple[float, str]] = []
        total = len(instantes)
        for i, t in enumerate(instantes):
            self._check_cancel()
            # Caché: el fotograma (vídeo, t) se reutiliza entre análisis.
            png = self._extractor.cached_frame(self._video, t)
            # Solo la disciplina: el contexto del usuario NO va por fotograma (el
            # marcador final se colaba en casi todas las anotaciones).
            # Se pasa el MAPA color→equipo (para ataque/defensa), no las
            # instrucciones del relato: por fotograma hacían alucinar alineaciones.
            prompt_vision = nts.vision_frame_prompt(self._sport, "", self._context)
            descripcion = self._vision.analyze_image(png, prompt_vision, self._model)
            # Datos estructurados de percepción pegados a la descripción: así el
            # informe (y la crónica) van más finos. Si hay percepción de
            # Comprensión, se reutiliza y NO se repiten YOLO/CLIP/OCR.
            resumen = self._perception_for(t, png)
            notas.append((t, nts.frame_annotation(descripcion, resumen)))
            if resumen:
                lecturas_vision.append((t, resumen))
            self.progress.emit(
                int((i + 1) / max(1, total) * 70), f"Analizando fotograma {i + 1}/{total}…"
            )

        # Marcador: se reutiliza la serie leída por Comprensión si viene; si no, se
        # lee por OCR cada ~2 s (separado de la visión). Sin OCR, la rejilla de visión.
        serie = self._serie_reutilizada()
        if serie is not None:
            marcador = sb.series_block(serie)
            juego_inicio = self._perception_juego_inicio
            descanso = self._perception_descanso
        else:
            lecturas_densas = self._read_scoreboard(duracion)
            lecturas = lecturas_densas if lecturas_densas else lecturas_vision
            serie = sb.score_series(lecturas)
            marcador = sb.marcador_block(lecturas)
            juego_inicio = sb.game_start(lecturas)
            descanso = sb.half_time(lecturas)
        transcript = self._perception_transcript or self._transcribe()
        self.progress.emit(80, "Componiendo el cuaderno…")
        self._check_cancel()
        def _prompt(max_anot: int, max_trans: int) -> str:
            return nts.analysis_prompt(
                self._video.name, duracion, notas, transcript,
                context=self._context, sport=self._sport, marcador=marcador,
                juego_inicio_s=juego_inicio, descanso_s=descanso,
                max_annotations=max_anot, max_transcript=max_trans,
                instructions=self._instructions,
            )

        # Se pide el cuaderno con el prompt completo; si el modelo se queda sin
        # espacio (o devuelve el JSON cortado), se reintenta UNA vez con la mitad de
        # anotaciones y menos transcripción. Con los «Datos del partido»
        # (alineación), los dorsales que devuelva el modelo pasan a nombre de pila.
        analysis = componer_cuaderno(
            lambda texto: self._chat.chat(
                [{"role": "user", "content": texto}], model=self._model
            ),
            [
                _prompt(nts.MAX_ANOTACIONES_CHARS, nts.MAX_TRANSCRIPT_CHARS),
                _prompt(nts.MAX_ANOTACIONES_CHARS // 2, 2000),
            ],
            parsear=lambda respuesta: nts.parse_analysis(
                respuesta, context=self._context, duration_s=duracion
            ),
            aviso=lambda _n: self.progress.emit(
                85, "Se agotó el espacio del modelo: reintentando con menos anotaciones…"
            ),
        )
        avisos = nts.analysis_coverage_warnings(
            analysis, len(sb.goals_from_series(serie))
        )
        for aviso in avisos:
            logger.warning("Cuaderno: %s", aviso)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        salida = self._out_dir / ANALYSIS_FILENAME
        salida.write_text(_analysis_to_json(analysis), encoding="utf-8")
        # Informe reutilizable: lo que la visión vio por fotograma y el audio. La
        # pestaña Crónica lo usa para escribir la crónica SIN volver a mirar el vídeo.
        informe = nts.informe_to_dict(
            video_path=str(self._video),
            video_name=self._video.name,
            duration_s=duracion,
            frames=notas,
            transcript=transcript,
            sport=self._sport,
            context=self._context,
        )
        # Marcador real y goles (derivados del OCR): quedan en el informe para que
        # la Crónica los use sin volver a mirar el vídeo.
        informe["marcador"] = [
            {"t_s": p.t_s, "izquierda": p.izquierda, "derecha": p.derecha}
            for p in serie
        ]
        informe["goles"] = [
            {"t_s": g.t_s, "lado": g.lado, "marcador": g.marcador}
            for g in sb.goals_from_series(serie)
        ]
        informe["juego_inicio"] = juego_inicio
        informe["descanso"] = descanso
        informe["avisos"] = avisos
        informe_path = nts.save_informe(
            self._out_dir / nts.INFORME_FILENAME, informe
        )
        self.progress.emit(100, "Análisis completado.")
        return {
            "analysis": analysis,
            "path": salida,
            "informe": informe_path,
            "marcador": sb.series_summary(serie),
            "avisos": avisos,
        }

    def _perception_engines(self) -> list:
        """Motores de percepción disponibles (objetos, etiquetas, OCR)."""
        candidatos = (self._objects_engine, self._labels_engine, self._ocr_engine)
        return [motor for motor in candidatos if motor is not None and motor.is_available()]

    def _perception_for(self, t: float, png: Path) -> str:
        """Percepción del fotograma: la de Comprensión (si hay) o los motores locales.

        Reutilizar la de Comprensión evita repetir YOLO/CLIP/OCR (y su coste) y
        mantiene el informe coherente con lo que ya vio esa pestaña.
        """
        if self._perception:
            return vu.nearest_annotation(t, self._perception)
        return self._perception_summary(png)

    def _perception_summary(self, png: Path) -> str:
        """Resumen de los motores de percepción para un fotograma (o cadena vacía)."""
        partes: list[str] = []
        for motor in self._perception_engines():
            self._check_cancel()
            prompt = self._clip_labels if getattr(motor, "id", "") == "clip" else ""
            try:
                resumen = motor.perceive(png, prompt).summary()
            except EngineError as exc:
                logger.debug(
                    "Cuaderno: percepción %s falló: %s", getattr(motor, "id", "?"), exc
                )
                continue
            if resumen:
                partes.append(resumen)
        return "\n".join(partes)

    def _serie_reutilizada(self):
        """Serie del marcador leída por Comprensión, o ``None`` si no hay ninguna."""
        if not self._perception_marcador:
            return None
        serie = []
        for punto in self._perception_marcador:
            try:
                serie.append(
                    sb.ScorePoint(
                        float(punto.get("t_s")),
                        int(punto.get("izquierda")),
                        int(punto.get("derecha")),
                    )
                )
            except (TypeError, ValueError, AttributeError):
                continue
        return serie or None

    def _read_scoreboard(self, duracion: float) -> list[tuple[float, str]]:
        """Lee el marcador por OCR con muestreo denso (cada ~2 s), separado de la visión.

        Devuelve ``[]`` si no hay OCR disponible (entonces ``work`` reutiliza las
        lecturas de los fotogramas de visión). El OCR se hace sobre un recorte de la
        franja superior, donde suele estar el marcador: es más rápido y evita
        confundir el tanteo con otros textos de la escena.
        """
        ocr = self._ocr_engine
        if ocr is None or not ocr.is_available():
            return []
        instantes = vu.scoreboard_timestamps(duracion)
        if not instantes:
            return []
        lecturas: list[tuple[float, str]] = []
        total = len(instantes)
        with tempfile.TemporaryDirectory(prefix="imperator_marcador_") as tmp:
            for indice, t in enumerate(instantes):
                self._check_cancel()
                png = Path(tmp) / f"marcador_{indice:05d}.png"
                try:
                    self._extractor.extract_cropped_frame(
                        self._video, t, png, crop=vu.SCOREBOARD_CROP
                    )
                    percepcion = ocr.perceive(png, "")
                    texto = (percepcion.text or "").strip()
                except EngineError as exc:
                    logger.debug("Cuaderno: marcador t=%gs falló: %s", t, exc)
                    continue
                if texto:
                    lecturas.append((t, texto))
                self.progress.emit(
                    70 + int((indice + 1) / max(1, total) * 3),
                    f"Leyendo marcador {indice + 1}/{total}…",
                )
        return lecturas

    def _transcribe(self) -> str:
        if not self._include_audio or self._whisper is None:
            return ""
        if not self._whisper.is_available():
            return ""
        with tempfile.TemporaryDirectory(prefix="imperator_notes_audio_") as tmp:
            wav = Path(tmp) / "audio.wav"
            self._check_cancel()
            self._extractor.extract_audio(self._video, wav)
            srt = self._whisper.transcribe(wav)
            try:
                return srt_to_text(Path(srt).read_text(encoding="utf-8", errors="replace"))
            except OSError:
                return ""


def _analysis_to_json(analysis: nts.Analysis) -> str:
    """Serializa el análisis a JSON (para reimportarlo después)."""

    def nodo(n: nts.StructureNode) -> dict:
        return {
            "titulo": n.titulo,
            "inicio_s": n.inicio_s,
            "fin_s": n.fin_s,
            "hijos": [nodo(h) for h in n.hijos],
        }

    data = {
        "titulo": analysis.titulo,
        "duracion_s": analysis.duracion_s,
        "lineas_narrativas": [
            {
                "nombre": linea.nombre,
                "color": linea.color,
                "eventos": [
                    {"t_s": e.t_s, "descripcion": e.descripcion} for e in linea.eventos
                ],
            }
            for linea in analysis.lineas
        ],
        "estructura": nodo(analysis.estructura) if analysis.estructura else None,
        "emocion": [
            {"t_s": p.t_s, "intensidad": p.intensidad, "etiqueta": p.etiqueta}
            for p in analysis.emocion
        ],
        "segmentos": [
            {
                "inicio_s": s.inicio_s,
                "fin_s": s.fin_s,
                "titulo": s.titulo,
                "resumen": s.resumen,
                "escenas": list(s.escenas),
            }
            for s in analysis.segmentos
        ],
        "notas": analysis.notas,
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


class NotesWidget(TaskWidget):
    """Pestaña Cuaderno: análisis cinematográfico local."""

    def __init__(self, engine_manager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._engine_manager = engine_manager
        self._prefs = prefs.PrefsStore(getattr(engine_manager, "config_manager", None))
        self._video: Path | None = None
        self._analysis: nts.Analysis | None = None
        self._extractor = FrameExtractor(cache=FrameCache())
        #: Carpeta de la última entrega (ZIP/análisis), para «Abrir carpeta».
        self._last_dir: Path | None = None
        #: Carpeta compartida de la sesión: el paquete y el análisis del MISMO
        #: vídeo deben quedar juntos (antes cada acción creaba su marca de tiempo
        #: y acababan en carpetas distintas, difíciles de encontrar).
        self._session_dir: Path | None = None
        #: Percepción estructurada de la pestaña Comprensión (comprension.json).
        #: Si está cargada y marcada, el análisis la reutiliza sin repetir motores.
        self._perception_data: dict | None = None
        self._build_ui()
        self._restore_instructions()
        # La disciplina es compartida: se restaura y se sincroniza en vivo.
        self._restore_discipline()
        discipline.BUS.changed.connect(self._on_sport_changed)

    def _restore_instructions(self) -> None:
        """Recupera las instrucciones guardadas (o pone el ejemplo de balonmano)."""
        guardado = self._prefs.get_str(prefs.NOTES_INSTRUCTIONS)
        self._txt_instructions.setPlainText(
            guardado if (guardado or "").strip() else sp.INSTRUCTIONS_EXAMPLE
        )

    # ---- UI --------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        fila_video = QHBoxLayout()
        self._btn_video = QPushButton("Elegir vídeo…")
        self._btn_video.clicked.connect(self._choose_video)
        self._lbl_video = QLabel("(sin vídeo)")
        fila_video.addWidget(self._btn_video)
        fila_video.addWidget(self._lbl_video, 1)
        root.addLayout(fila_video)

        fila_opts = QHBoxLayout()
        fila_opts.addWidget(QLabel("Fotogramas/s:"))
        self._spn_fps = QSpinBox()
        self._spn_fps.setRange(1, 10)
        self._spn_fps.setValue(int(nts.DEFAULT_FPS))
        fila_opts.addWidget(self._spn_fps)
        fila_opts.addWidget(QLabel("Máx. paquete:"))
        self._spn_max = QSpinBox()
        self._spn_max.setRange(1, nts.MAX_MAX_FRAMES)
        self._spn_max.setValue(nts.DEFAULT_MAX_FRAMES)
        fila_opts.addWidget(self._spn_max)
        fila_opts.addWidget(QLabel("Visión:"))
        self._spn_vision = QSpinBox()
        self._spn_vision.setRange(4, nts.MAX_MAX_FRAMES)
        self._spn_vision.setValue(nts.DEFAULT_VISION_FRAMES)
        self._spn_vision.setToolTip(
            "Fotogramas que mira la IA de visión, repartidos por TODO el vídeo.\n"
            "Más fotogramas = más detalle y más tiempo. Para un partido largo,\n"
            "súbelo a 40-60 si quieres cubrir más momentos (24 = uno cada ~2 min)."
        )
        fila_opts.addWidget(self._spn_vision)
        self._chk_audio = QCheckBox("Transcribir audio (Whisper)")
        self._chk_audio.setChecked(True)
        fila_opts.addWidget(self._chk_audio)
        fila_opts.addStretch(1)
        root.addLayout(fila_opts)

        self._cmb_sport = QComboBox()
        discipline.fill(self._cmb_sport)
        self._cmb_sport.setToolTip(
            "Disciplina del vídeo. Se le pasan sus reglas y vocabulario a la IA, que\n"
            "si no confunde deportes (p. ej. balonmano con voleibol). Es la MISMA\n"
            "disciplina en Cuaderno, Crónica y Comprensión: cambiarla aquí la cambia\n"
            "en las demás."
        )
        self._cmb_sport.currentIndexChanged.connect(self._on_sport_picked)
        fila_sport = QHBoxLayout()
        fila_sport.addWidget(QLabel("Disciplina:"))
        fila_sport.addWidget(self._cmb_sport)
        fila_sport.addStretch(1)
        root.addLayout(fila_sport)

        root.addWidget(
            QLabel(
                "Contexto (equipos, colores, marcador, incidencias; la IA lo toma "
                "como cierto):"
            )
        )
        self._txt_context = QPlainTextEdit()
        self._txt_context.setPlaceholderText(nts.CONTEXT_EXAMPLE)
        self._txt_context.setFixedHeight(72)
        root.addWidget(self._txt_context)

        # Instrucciones para la IA: qué quiere el usuario que cuente el cuaderno (y,
        # de ahí, la crónica). Van al prompt de cada fotograma y al del cuaderno.
        fila_instr = QHBoxLayout()
        fila_instr.addWidget(QLabel("Instrucciones para la IA (qué quieres contar):"))
        fila_instr.addStretch(1)
        self._btn_ejemplo_ia = QPushButton("Poner ejemplo (balonmano)")
        self._btn_ejemplo_ia.setToolTip(
            "Rellena el ejemplo recomendado: presentar la alineación al principio\n"
            "(mientras calientan) y después centrarse en goles, paradas del portero\n"
            "y lanzamientos de 7 metros. Edítalo a tu gusto."
        )
        self._btn_ejemplo_ia.clicked.connect(
            lambda: self._txt_instructions.setPlainText(sp.INSTRUCTIONS_EXAMPLE)
        )
        fila_instr.addWidget(self._btn_ejemplo_ia)
        root.addLayout(fila_instr)
        self._txt_instructions = QPlainTextEdit()
        self._txt_instructions.setFixedHeight(72)
        self._txt_instructions.setPlaceholderText(sp.INSTRUCTIONS_EXAMPLE)
        self._txt_instructions.setToolTip(
            "Escribe lo que quieres que cuente el análisis y en qué orden. Se pasa a\n"
            "la visión de cada fotograma y al texto del cuaderno."
        )
        root.addWidget(self._txt_instructions)

        # Señales de percepción opcionales: añaden objetos/etiquetas/OCR a cada
        # fotograma (además de la descripción de la visión). Así el cuaderno —y el
        # informe que consume la Crónica— son más exactos.
        fila_senales = QHBoxLayout()
        # Por defecto, las tres señales activadas: es la opción de más calidad.
        self._chk_objects = QCheckBox("Objetos (YOLO)")
        self._chk_objects.setChecked(True)
        self._chk_objects.setToolTip("Añade a cada fotograma los objetos detectados (YOLO).")
        self._chk_labels = QCheckBox("Etiquetas (CLIP)")
        self._chk_labels.setChecked(True)
        self._chk_labels.setToolTip("Añade etiquetas de CLIP según el vocabulario de abajo.")
        self._chk_ocr = QCheckBox("Texto en pantalla (OCR)")
        self._chk_ocr.setChecked(True)
        self._chk_ocr.setToolTip(
            "Añade el texto leído en pantalla (marcador, reloj, carteles). Es la\n"
            "fuente fiable del marcador y del tiempo: la visión ya no los inventa."
        )
        for _senal in (self._chk_objects, self._chk_labels, self._chk_ocr):
            fila_senales.addWidget(_senal)
        fila_senales.addStretch(1)
        root.addLayout(fila_senales)

        fila_clip = QHBoxLayout()
        fila_clip.addWidget(QLabel("Etiquetas CLIP:"))
        self._txt_clip_labels = QLineEdit()
        self._txt_clip_labels.setText(clip_labels_text())
        self._txt_clip_labels.setToolTip(
            "Vocabulario que puntúa CLIP (lista separada por comas). CLIP no describe:\n"
            "elige entre estas etiquetas. Lo recomendado: etiquetas CONCRETAS y\n"
            "separables del contenido (acciones y estados del juego), sin nombres ni\n"
            "colores (esos los pone la alineación). Genéricas («calle, coche, animal»)\n"
            "o dos que significan casi lo mismo dan ruido."
        )
        fila_clip.addWidget(self._txt_clip_labels, 1)
        root.addLayout(fila_clip)

        # Reutilizar la percepción que ya hizo la pestaña Comprensión: evita
        # repetir YOLO/CLIP/OCR (y su coste) y mantiene el informe coherente.
        fila_comp = QHBoxLayout()
        self._btn_comp = QPushButton("Usar percepción de Comprensión…")
        self._btn_comp.setToolTip(
            "Carga el comprension.json que dejó la pestaña Comprensión y reutiliza\n"
            "su percepción (objetos/etiquetas/OCR por fotograma) y su transcripción,\n"
            "sin volver a analizar el vídeo en los motores de percepción."
        )
        self._btn_comp.clicked.connect(self._load_comprension_dialog)
        self._chk_comp = QCheckBox("Usar esa percepción (no repetir YOLO/CLIP/OCR)")
        self._chk_comp.setEnabled(False)
        self._lbl_comp = QLabel("(sin percepción de Comprensión)")
        fila_comp.addWidget(self._btn_comp)
        fila_comp.addWidget(self._chk_comp)
        fila_comp.addWidget(self._lbl_comp, 1)
        root.addLayout(fila_comp)

        # Ayuda a la Crónica: el informe.json que deja «Analizar en local» es lo que
        # la Crónica carga para escribir sin volver a mirar el vídeo.
        self._lbl_ayuda_cronica = QLabel(
            "«Analizar en local» guarda informe.json: la pestaña Crónica lo carga "
            "para escribir la crónica sin volver a analizar el vídeo."
        )
        self._lbl_ayuda_cronica.setWordWrap(True)
        root.addWidget(self._lbl_ayuda_cronica)

        fila_botones = QHBoxLayout()
        self._btn_package = QPushButton("Generar paquete (ZIP)…")
        self._btn_package.clicked.connect(self._package)
        self._btn_analyze = QPushButton("Analizar en local (Ollama)")
        self._btn_analyze.clicked.connect(self._analyze)
        self._btn_import = QPushButton("Importar análisis (JSON)…")
        self._btn_import.clicked.connect(self._import)
        self._btn_export = QPushButton("Exportar Markdown…")
        self._btn_export.clicked.connect(self._export)
        self._btn_open = QPushButton("Abrir carpeta")
        self._btn_open.setEnabled(False)
        self._btn_open.setToolTip(
            "Abre la carpeta donde quedan el paquete (ZIP) y el análisis\n"
            "(analisis.json): projects/notes/<vídeo>_<fecha>/."
        )
        self._btn_open.clicked.connect(self._open_folder)
        self._btn_cancel = QPushButton("Cancelar")
        self._btn_cancel.clicked.connect(self._cancel)
        for b in (
            self._btn_package, self._btn_analyze, self._btn_import,
            self._btn_export, self._btn_open, self._btn_cancel,
        ):
            fila_botones.addWidget(b)
        fila_botones.addStretch(1)
        root.addLayout(fila_botones)

        self._status = QLabel("")
        root.addWidget(self._status)

        # Contraste del contexto del usuario con lo que ve/oye el vídeo (avisos,
        # no correcciones): determinista, sin llamadas extra al modelo.
        self._lbl_contraste = QLabel("")
        self._lbl_contraste.setWordWrap(True)
        self._lbl_contraste.setStyleSheet(f"color: {semantic('warning_fg').name()};")
        root.addWidget(self._lbl_contraste)

        # Marcador real y goles leídos por OCR (vacío si el OCR no está activo).
        self._lbl_marcador = QLabel("")
        self._lbl_marcador.setWordWrap(True)
        self._lbl_marcador.setStyleSheet(f"color: {semantic('ok_fg').name()};")
        root.addWidget(self._lbl_marcador)
        # Barra de avance + reloj, como en las demás tareas (antes no había
        # forma de ver si un paquete o un análisis seguían vivos).
        self._montar_progreso(root)

        self._title = QLabel("")
        self._title.setStyleSheet("font-weight: bold;")
        root.addWidget(self._title)

        self._tabs = QTabWidget()
        self._table_lanes = QTableWidget(0, 3)
        self._table_lanes.setHorizontalHeaderLabels(["Línea", "Momento", "Descripción"])
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Estructura narrativa", "Inicio", "Fin"])
        self._curve = EmotionCurveWidget()
        self._table_segments = QTableWidget(0, 4)
        self._table_segments.setHorizontalHeaderLabels(["Inicio", "Fin", "Título", "Resumen"])
        self._tabs.addTab(self._table_lanes, "Líneas narrativas")
        self._tabs.addTab(self._tree, "Estructura")
        self._tabs.addTab(self._curve, "Emoción")
        self._tabs.addTab(self._table_segments, "Segmentos")
        root.addWidget(self._tabs, 1)

        self._btn_cancel.setEnabled(False)

    # ---- Controles de estado --------------------------------------

    def _busy_widgets(self) -> tuple:
        return (
            self._btn_video, self._btn_package, self._btn_analyze,
            self._btn_import, self._btn_export, self._spn_fps, self._spn_max,
            self._txt_instructions, self._btn_ejemplo_ia,
            self._spn_vision, self._chk_audio, self._cmb_sport, self._txt_context,
            self._chk_objects, self._chk_labels, self._chk_ocr, self._txt_clip_labels,
            self._btn_comp, self._chk_comp,
        )

    def _busy_only_widgets(self) -> tuple:
        return (self._btn_cancel,)

    # ---- Acciones --------------------------------------------------

    def _choose_video(self) -> None:
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Elegir vídeo", "", VIDEO_FILTER_ALL
        )
        if ruta:
            self._video = Path(ruta)
            self._lbl_video.setText(self._video.name)
            # Vídeo nuevo -> carpeta de sesión nueva (no mezclar entregas).
            self._session_dir = None
            self._lbl_contraste.clear()
            self._lbl_marcador.clear()

    def _on_sport_picked(self, _index: int) -> None:
        """Guarda la disciplina elegida y la propaga a las demás pestañas."""
        clave = self._cmb_sport.currentData()
        if not clave:
            return
        self._prefs.set(prefs.SPORT, clave)
        discipline.BUS.changed.emit(str(clave))

    def _on_sport_changed(self, clave: str) -> None:
        """Aplica la disciplina elegida en otra pestaña (sin reemitir)."""
        discipline.apply(self._cmb_sport, str(clave))

    def _restore_discipline(self) -> None:
        discipline.apply(self._cmb_sport, discipline.saved(self._prefs))

    def _require_video(self) -> Path | None:
        if self._video is None:
            self._set_status("Elige primero un vídeo.")
            return None
        return self._video

    def _out_dir(self) -> Path:
        """Carpeta del paquete y del análisis (una sola por vídeo y sesión)."""
        destino = self._session_dir
        if destino is None:
            marca = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = self._video.stem if self._video else "cuaderno"
            destino = NOTES_DIR / f"{stem}_{marca}"
            self._session_dir = destino
        destino.mkdir(parents=True, exist_ok=True)
        return destino

    def _package(self) -> None:
        video = self._require_video()
        if video is None:
            return
        worker = PackageWorker(
            video,
            extractor=self._extractor,
            fps=float(self._spn_fps.value()),
            max_frames=self._spn_max.value(),
            include_audio=self._chk_audio.isChecked(),
            whisper_engine=self._engine_manager.get_engine("whisper"),
            out_dir=self._out_dir(),
            sport=self._cmb_sport.currentData(),
            context=self._txt_context.toPlainText(),
        )
        self._start_worker(worker, self._on_package_done)

    def _analyze(self) -> None:
        video = self._require_video()
        if video is None:
            return
        instructions = self._txt_instructions.toPlainText()
        self._prefs.set(prefs.NOTES_INSTRUCTIONS, instructions)
        # Si hay percepción de Comprensión cargada y marcada, se reutiliza en vez
        # de volver a pasar los motores por cada fotograma.
        percepcion = None
        transcript_comp = ""
        marcador_comp: list = []
        juego_inicio_comp = None
        descanso_comp = None
        if self._perception_data is not None and self._chk_comp.isChecked():
            percepcion = vu.understanding_frames(self._perception_data)
            transcript_comp = str(self._perception_data.get("transcripcion", ""))
            marcador_comp = list(self._perception_data.get("marcador", []) or [])
            juego_inicio_comp = self._perception_data.get("juego_inicio")
            descanso_comp = self._perception_data.get("descanso")
        worker = AnalyzeWorker(
            video,
            extractor=self._extractor,
            vision_engine=self._engine_manager.get_engine("ollama_vision"),
            chat_engine=ce.preferred_chat_engine(self._engine_manager),
            whisper_engine=self._engine_manager.get_engine("whisper"),
            vision_frames=self._spn_vision.value(),
            include_audio=self._chk_audio.isChecked(),
            model=None,
            out_dir=self._out_dir(),
            sport=self._cmb_sport.currentData(),
            context=self._txt_context.toPlainText(),
            instructions=instructions,
            objects_engine=self._engine_manager.get_engine("yolo")
            if self._chk_objects.isChecked()
            else None,
            labels_engine=self._engine_manager.get_engine("clip")
            if self._chk_labels.isChecked()
            else None,
            ocr_engine=self._engine_manager.get_engine("paddle_ocr")
            if self._chk_ocr.isChecked()
            else None,
            clip_labels=self._txt_clip_labels.text(),
            perception=percepcion,
            perception_transcript=transcript_comp,
            perception_marcador=marcador_comp,
            perception_juego_inicio=juego_inicio_comp,
            perception_descanso=descanso_comp,
        )
        self._start_worker(worker, self._on_analysis_done)

    def _load_comprension_dialog(self) -> None:
        """Elige el ``comprension.json`` que dejó la pestaña Comprensión."""
        ruta, _ = QFileDialog.getOpenFileName(
            self,
            "Percepción de Comprensión",
            str(UNDERSTANDING_DIR),
            "Informe (*.json);;Todos (*)",
        )
        if ruta:
            self._load_comprension(ruta)

    def _load_comprension(self, ruta) -> None:
        """Carga la percepción de Comprensión; el análisis la reutilizará."""
        try:
            data = vu.load_understanding(ruta)
        except vu.UnderstandingError as exc:
            self._set_status(f"Error: {exc}")
            return
        self._perception_data = data
        self._chk_comp.setEnabled(True)
        self._chk_comp.setChecked(True)
        # Si el «Contexto» está vacío, se rellena con el que traía la percepción de
        # Comprensión (fecha, lugar, equipos, alineación): así el Cuaderno no lo
        # pierde al reutilizarla y la Crónica lo hereda del informe.json.
        contexto_comp = str(data.get("contexto") or "").strip()
        if contexto_comp and not self._txt_context.toPlainText().strip():
            self._txt_context.setPlainText(contexto_comp)
        nombre = data.get("video_name") or Path(ruta).name
        self._lbl_comp.setText(
            f"{nombre} ({len(vu.understanding_frames(data))} fotogramas)"
        )
        self._mostrar_contraste(data)
        self._set_status(
            "Percepción de Comprensión cargada: el análisis la reutilizará "
            "(no se repetirán YOLO/CLIP/OCR)."
        )

    def _import(self) -> None:
        ruta, _ = QFileDialog.getOpenFileName(self, "Importar análisis", "", "JSON (*.json);;Todos (*)")
        if not ruta:
            return
        try:
            texto = Path(ruta).read_text(encoding="utf-8", errors="replace")
            analysis = nts.parse_analysis(texto, context=self._txt_context.toPlainText())
        except (OSError, nts.NotesError) as exc:
            self._set_status(f"Error al importar: {exc}")
            return
        self._show_analysis(analysis)
        self._set_status("Análisis importado.")

    def _export(self) -> None:
        if self._analysis is None:
            self._set_status("No hay análisis que exportar.")
            return
        ruta, _ = QFileDialog.getSaveFileName(self, "Exportar Markdown", "cuaderno.md", "Markdown (*.md)")
        if not ruta:
            return
        try:
            Path(ruta).write_text(nts.build_markdown(self._analysis), encoding="utf-8")
        except OSError as exc:
            self._set_status(f"Error al exportar: {exc}")
            return
        self._set_status(f"Exportado a {ruta}")

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.requestInterruption()

    # ---- Ciclo del worker -----------------------------------------

    def _start_worker(self, worker: ThreadWorker, on_result) -> None:
        self._worker = worker
        worker.progress.connect(self._on_progress)
        worker.result.connect(on_result)
        worker.failed.connect(self._on_error)
        worker.finished.connect(self._on_finished)
        self._set_busy(True)
        self._empezar_progreso()
        self._set_status("Trabajando…")
        worker.start()

    def _on_progress(self, pct: int, msg: str) -> None:
        self._barra.setRange(0, 100)
        self._barra.setValue(max(0, min(100, int(pct))))
        self._set_status(f"{msg} ({pct}%)")

    def _on_package_done(self, data: dict) -> None:
        self._recordar_carpeta(Path(data["zip"]).parent)
        self._set_status(
            f"Paquete listo: {Path(data['zip']).name} ({data['frames']} fotogramas)"
            + (" con transcripción." if data.get("transcript") else ".")
            + f" · en {Path(data['zip']).parent}"
        )

    def _on_analysis_done(self, data: dict) -> None:
        self._show_analysis(data["analysis"])
        self._recordar_carpeta(Path(data["path"]).parent)
        self._mostrar_marcador(data.get("marcador", ""))
        self._mostrar_contraste_desde_informe(data.get("informe"))
        self._set_status(
            f"Análisis listo: {Path(data['path']).name} "
            "(informe.json guardado para la pestaña Crónica)."
        )

    def _mostrar_marcador(self, texto: str) -> None:
        """Muestra (o limpia) el resumen del marcador leído por OCR."""
        self._lbl_marcador.setText(str(texto or ""))

    def _mostrar_contraste_desde_informe(self, informe_path) -> None:
        """Contrasta el contexto con el informe.json recién guardado (si lo hay)."""
        if not informe_path:
            return
        try:
            data = nts.load_informe(informe_path)
        except nts.NotesError:
            return
        self._mostrar_contraste(data)

    def _mostrar_contraste(self, data: dict | None) -> None:
        """Pinta los avisos (o «sin avisos») del contexto frente a la percepción."""
        if not data:
            self._lbl_contraste.clear()
            return
        avisos = nts.contrast_context(self._txt_context.toPlainText(), data)
        if not avisos:
            self._lbl_contraste.setText("Contraste de contexto: sin avisos.")
        else:
            self._lbl_contraste.setText(
                "Contraste de contexto:\n- " + "\n- ".join(avisos)
            )

    def _recordar_carpeta(self, carpeta: Path) -> None:
        """Guarda la carpeta de la entrega y habilita «Abrir carpeta»."""
        self._last_dir = Path(carpeta)
        self._btn_open.setEnabled(True)

    def _open_folder(self) -> None:
        """Abre en el Explorador la carpeta con el paquete y el análisis."""
        destino = self._last_dir or self._session_dir or NOTES_DIR
        if not Path(destino).is_dir():
            self._set_status(f"La carpeta todavía no existe: {destino}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(destino))))
        self._set_status(f"Carpeta abierta: {destino}")

    # ---- Render ----------------------------------------------------

    def _show_analysis(self, analysis: nts.Analysis) -> None:
        self._analysis = analysis
        self._title.setText(analysis.titulo or "Cuaderno de análisis")
        self._fill_lanes(analysis)
        self._fill_tree(analysis)
        self._curve.set_points(analysis.emocion)
        self._fill_segments(analysis)

    def _fill_lanes(self, analysis: nts.Analysis) -> None:
        filas = [
            (linea.nombre, e.label, e.descripcion)
            for linea in analysis.lineas
            for e in linea.eventos
        ]
        self._table_lanes.setRowCount(len(filas))
        for i, (nombre, momento, desc) in enumerate(filas):
            for j, valor in enumerate((nombre, momento, desc)):
                self._table_lanes.setItem(i, j, QTableWidgetItem(str(valor)))

    def _fill_tree(self, analysis: nts.Analysis) -> None:
        self._tree.clear()
        if analysis.estructura is None:
            return

        def añadir(nodo: nts.StructureNode, padre) -> None:
            item = QTreeWidgetItem([
                nodo.titulo,
                nts.format_timestamp(nodo.inicio_s) if nodo.inicio_s is not None else "",
                nts.format_timestamp(nodo.fin_s) if nodo.fin_s is not None else "",
            ])
            if padre is None:
                self._tree.addTopLevelItem(item)
            else:
                padre.addChild(item)
            for hijo in nodo.hijos:
                añadir(hijo, item)

        añadir(analysis.estructura, None)
        self._tree.expandAll()

    def _fill_segments(self, analysis: nts.Analysis) -> None:
        segs = analysis.segmentos
        self._table_segments.setRowCount(len(segs))
        for i, seg in enumerate(segs):
            resumen = seg.resumen or ""
            if seg.escenas:
                resumen = (resumen + " · " if resumen else "") + "; ".join(seg.escenas)
            valores = (nts.format_timestamp(seg.inicio_s), nts.format_timestamp(seg.fin_s), seg.titulo, resumen)
            for j, valor in enumerate(valores):
                self._table_segments.setItem(i, j, QTableWidgetItem(str(valor)))

    def set_engine_manager(self, engine_manager) -> None:
        """Relee los motores tras cambiar la configuración."""
        self._engine_manager = engine_manager
