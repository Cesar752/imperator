"""Pestaña **Crónica**: convierte un partido en un vídeo comentado por IA.

Encadena lo que ya existía suelto en la app, sin inventar motores nuevos:

1. **Mira el vídeo**: describe fotogramas con la visión local (Ollama) y, si se
   pide, transcribe el audio con Whisper.
2. **Escribe la crónica**: el chat (el motor elegido en Chat) compone frases
   cortas **con el segundo en el que se dicen**.
3. **La locuta**: cada frase se sintetiza con Piper/Kokoro/XTTS y se coloca en su
   segundo con ``adelay``; todo se mezcla en una sola pista (``amix``).
4. **Le pone música**: opcional, con MusicGen.
5. **Monta el vídeo**: subtítulos quemados + voz + música (con *ducking*) mediante
   el Ensamblador, y deja una ``*_cronica.md`` con el texto y sus tiempos.

La lógica pura (prompt, parseo, SRT, grafo de la voz) vive en
:mod:`services.chronicle_params`; aquí sólo se ejecuta.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import wave
from pathlib import Path

from PyQt6 import sip
from PyQt6.QtCore import QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.frame_cache import FrameCache
from core.frame_extractor import FrameExtractor
from core.paths import OUTPUT, PROJECT_ROOT
from core.subprocess_utils import no_window_kwargs
from core.video_assembler import VideoAssembler
from engines.base_engine import CANCEL_MESSAGE, EngineError
from services import acta_params as acp
from services import assembler_params as asp
from services import audio_params as ap
from services import chat_engine as ce
from services import chronicle_params as cp
from services import notes_params as nts
from services import preferences as prefs
from services import sport_params as sp
from services import video_enhance_params as vep
from services import video_understanding as vu
from ui import discipline
from ui.task_widget import TaskWidget
from ui.workers import ThreadWorker, gpu_slot, run_pool_task

# Import diferido y protegido: sin PyMuPDF el acta en PDF no se puede leer, pero la
# pestaña (y el resto de la app) siguen funcionando.
try:
    import pymupdf
except ImportError:  # pragma: no cover - dependencia declarada en requirements.txt
    pymupdf = None

logger = logging.getLogger(__name__)

#: Carpeta donde el Cuaderno deja sus informes (informe.json).
NOTES_DIR = PROJECT_ROOT / "projects" / "notes"
#: Carpeta donde Comprensión deja su percepción estructurada (comprension.json).
PERCEPTION_DIR = PROJECT_ROOT / "projects" / "comprension"

#: Timeout del pase de ffmpeg que mezcla la voz (una frase por pista).
MIX_TIMEOUT = 900
#: Resolución con la que se rasteriza una página escaneada del acta antes del OCR.
ACTA_OCR_DPI = 200


class ChronicleWorker(ThreadWorker):
    """Ejecuta la crónica completa (análisis → guion → voz → música → montaje)."""

    def __init__(
        self,
        video: Path,
        *,
        extractor: FrameExtractor,
        vision_engine=None,
        chat_engine=None,
        whisper_engine=None,
        audio_engine=None,
        music_engine=None,
        output_dir: Path | None = None,
        model: str | None = None,
        vision_model: str | None = None,
        chat_model: str | None = None,
        voice: str | None = None,
        voice_speed: float = 1.0,
        language: str = cp.DEFAULT_LANGUAGE,
        max_lines: int = cp.DEFAULT_MAX_LINES,
        every_s: float = 5.0,
        max_frames: int = 16,
        transcribe: bool = True,
        music_prompt: str = "",
        music_volume: float = 0.25,
        voice_volume: float = 1.0,
        original_volume: float = cp.DEFAULT_ORIGINAL_VOLUME,
        keep_original: bool = True,
        subtitle_style: dict | None = None,
        context: str = "",
        sport: str = "",
        instructions: str = "",
        report: str = "",
        scheduler=None,
        liberar_otros=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._video = Path(video)
        self._extractor = extractor
        self._vision = vision_engine
        self._chat = chat_engine
        self._whisper = whisper_engine
        self._audio = audio_engine
        self._music = music_engine
        self._output_dir = Path(output_dir) if output_dir else OUTPUT
        self._model = model
        #: La Crónica puede usar una IA distinta (mejor) que el resto de la app: se
        #: permiten modelos separados para la visión (mira los fotogramas) y el chat
        #: (escribe el guion). Si no se indican, ``model`` vale para los dos.
        self._vision_model = vision_model if vision_model is not None else model
        self._chat_model = chat_model if chat_model is not None else model
        self._voice = voice
        self._voice_speed = float(voice_speed)
        self._language = language
        self._max_lines = max_lines
        self._every_s = float(every_s)
        self._max_frames = int(max_frames)
        self._transcribe = bool(transcribe)
        self._music_prompt = (music_prompt or "").strip()
        self._music_volume = float(music_volume)
        self._voice_volume = float(voice_volume)
        #: El ambiente va de fondo: al 100 % el pabellón tapa al locutor.
        self._original_volume = float(original_volume)
        #: Por defecto se **conserva** el sonido de la grabación bajo la voz: es lo
        #: que se espera al comentar un partido (se oye el público y el balón).
        self._keep_original = bool(keep_original)
        self._style = subtitle_style
        #: Datos del partido que aporta el usuario (equipos, colores, alineación,
        #: marcador). Sin ellos la IA ve colores; con ellos, nombres de verdad.
        self._context = str(context or "")
        #: Disciplina (reglas y vocabulario). Sin ella la IA confunde deportes.
        self._sport = str(sport or "")
        #: Instrucciones del usuario (qué contar y con qué prioridad): van al prompt
        #: de cada fotograma y al que escribe la crónica.
        self._instructions = str(instructions or "")
        #: Informe ya hecho (del Cuaderno): si llega, NO se vuelve a mirar el vídeo.
        self._report = str(report or "").strip()
        self._scheduler = scheduler
        self._liberar_otros = liberar_otros
        self._duration: float | None = None
        #: Cuántas frases iniciales son la apertura (sede, fecha y alineación de los
        #: dos equipos): se locutan encadenadas por su duración REAL de audio.
        self._opening_count = 0
        #: Avisos de verificación (goles/paradas que la crónica pudo saltarse).
        self._avisos: list[str] = []

    # ---- utilidades -------------------------------------------------

    def _check_cancel(self) -> None:
        if self.isInterruptionRequested():
            raise EngineError(CANCEL_MESSAGE)

    def _require(self, engine, mensaje: str) -> None:
        if engine is None or not engine.is_available():
            raise EngineError(mensaje)

    # ---- pipeline ---------------------------------------------------

    def work(self) -> str:
        # Una sola reserva de GPU para todo el trabajo: visión, voz y música usan
        # la misma tarjeta y no tiene sentido pelearse consigo mismo por ella.
        with gpu_slot(
            self._scheduler, "understanding", liberar_otros=self._liberar_otros
        ):
            informe = self._informe_o_analisis()
            self._check_cancel()
            lineas = self._write_chronicle(informe)
            for aviso in self._avisos:
                logger.warning("Crónica: %s", aviso)
            self._check_cancel()
            narracion, narradas = self._narrate(lineas)
            self._check_cancel()
            musica = self._generate_music()
            self._check_cancel()
            return str(self._assemble(narradas, narracion, musica))

    # ---- 1. análisis (o informe ya hecho) ---------------------------

    def _informe_o_analisis(self) -> str:
        """Informe de la crónica: el del Cuaderno si se cargó, o el análisis propio.

        Con informe, se reutiliza lo que el Cuaderno ya «vio» (descripciones por
        fotograma + audio) y **no se vuelve a analizar el vídeo**: ni visión ni
        Whisper. Solo hace falta la duración para colocar bien los tiempos.
        """
        if self._report:
            self.progress.emit(
                40, "Usando el informe del Cuaderno (sin volver a mirar el vídeo)…"
            )
            duracion = self._extractor.duration(self._video)
            if duracion is None or duracion <= 0:
                raise EngineError("No se pudo leer la duración del vídeo (¿está dañado?).")
            self._duration = duracion
            return self._report
        return self._analyze()

    def _analyze(self) -> str:
        self._require(
            self._vision,
            "No hay motor de visión (Ollama Vision) para describir el vídeo.",
        )
        duracion = self._extractor.duration(self._video)
        if duracion is None or duracion <= 0:
            raise EngineError("No se pudo leer la duración del vídeo (¿está dañado?).")
        self._duration = duracion

        instantes = vu.sample_timestamps(duracion, self._every_s, self._max_frames)
        # La disciplina aporta las reglas; los datos del partido, los colores: así
        # la visión dice «el jugador azul» y no confunde el deporte.
        # Las instrucciones del usuario son para el GUION (qué contar y en qué
        # orden), no para cada fotograma: pasarlas aquí hacía que el modelo
        # intentara «presentar la alineación» en todos y alucinara listas enormes.
        prompt_vision = cp.vision_frame_prompt(self._context, self._sport)
        notas: list[tuple[float, str]] = []
        total = max(1, len(instantes))
        for indice, t_s in enumerate(instantes):
            self._check_cancel()
            # Caché: repetir la crónica no vuelve a decodificar los fotogramas.
            png = self._extractor.cached_frame(self._video, t_s)
            descripcion = self._vision.analyze_image(
                png, prompt_vision, self._vision_model
            )
            notas.append((t_s, descripcion))
            self.progress.emit(
                int((indice + 1) / total * 40), f"Fotograma {indice + 1}/{total}"
            )

        partes = [
            "LO QUE SE VE EN EL VÍDEO (una línea por instante analizado):",
            cp.frame_notes_text(notas) or "(no se pudo describir ningún fotograma)",
        ]
        # La transcripción se acota: la de un partido largo no cabía en ``num_ctx``
        # y el modelo devolvía vacío («se agotó el espacio de la respuesta»).
        transcripcion = cp.clamp_transcript(self._transcribe_audio(duracion))
        if transcripcion:
            partes += ["", "AUDIO DEL VÍDEO (transcripción literal):", transcripcion]
        return "\n".join(partes)

    def _transcribe_audio(self, duracion: float) -> str:
        """Transcribe el audio del vídeo (opcional; sin Whisper no es un fallo)."""
        if not self._transcribe:
            return ""
        if self._whisper is None or not self._whisper.is_available():
            logger.info("Crónica: sin Whisper; se escribe solo desde la imagen.")
            return ""
        with tempfile.TemporaryDirectory(prefix="imperator_cronica_") as tmp:
            wav = Path(tmp) / "audio.wav"
            self.progress.emit(42, "Transcribiendo el audio…")
            try:
                self._extractor.extract_audio(self._video, wav)
                srt = self._whisper.transcribe(wav, language=self._language or None)
                return vu.srt_to_text(Path(srt).read_text(encoding="utf-8", errors="replace"))
            except EngineError as exc:
                # El audio se pierde, pero la crónica se puede escribir igual: se
                # avisa y se sigue en vez de abortar el trabajo completo.
                logger.warning("Crónica: no se pudo transcribir el audio: %s", exc)
                return ""

    # ---- 2. guion ---------------------------------------------------

    def _write_chronicle(self, informe: str):
        self._require(
            self._chat,
            "No hay motor de chat disponible para escribir la crónica "
            "(elige uno en la pestaña Chat → Motores).",
        )
        self.progress.emit(50, "Escribiendo la crónica…")
        # La alineación puede venir de la caja o, si está vacía, del propio informe
        # (el Cuaderno la incluye): en ambos casos se traduce el dorsal al nombre.
        contexto = self._context or cp.datos_del_informe(informe)
        prompt = cp.chronicle_prompt(
            informe,
            language=self._language,
            max_lines=self._max_lines,
            duration_s=self._duration,
            video_name=self._video.name,
            context=self._context,
            sport=self._sport,
            instructions=self._instructions,
        )
        respuesta = self._chat.chat(
            [{"role": "user", "content": prompt}], self._chat_model
        )
        lineas = cp.parse_chronicle(
            respuesta,
            duration=self._duration,
            max_lines=self._max_lines,
            context=contexto,
        )
        # Anclar cada frase al instante anotado (misma rejilla que Comprensión y
        # Cuaderno): así la crónica no va a destiempo respecto al hecho narrado.
        instantes = cp.report_timestamps(informe)
        # Se coloca cada frase en SU momento: se corrige la deriva sólo si hay una
        # anotación muy cerca (±3 s) y se respeta el resto, para no desfasar la
        # narración (antes una frase del calentamiento podía saltar al 3:40).
        colocadas = cp.place_lines(lineas, instantes)
        apertura = cp.opening_lines(contexto)
        self._opening_count = len(apertura)
        # Apertura SIEMPRE al principio (sede, fecha y alineación de los DOS
        # equipos), compuesta con reglas: no depende de que el modelo la escriba ni
        # de que exista una anotación [MM:SS] en ese momento.
        finales = cp.prepend_opening(apertura, colocadas)
        self._avisos = cp.coverage_warnings(informe, finales)
        return finales

    # ---- 3. voz -----------------------------------------------------

    def _narrate(self, lineas) -> tuple[Path, list[cp.ChronicleLine]]:
        self._require(self._audio, "No hay motor de voz (Piper/Kokoro/XTTS) disponible.")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        destino = self._output_dir / f"{cp.default_output_name(self._video)}_voz.wav"
        narradas: list[cp.ChronicleLine] = []
        with tempfile.TemporaryDirectory(prefix="imperator_cronica_") as tmp:
            entries: list[tuple[float, object]] = []
            saltadas = 0
            total = max(1, len(lineas))
            # La apertura (sede, fecha y las dos alineaciones) se encadena por la
            # duración REAL del audio ya sintetizado: estimar la duración a ojo hacía
            # que la alineación del Muskiz se pisara con la de los Corazonistas.
            cursor = 0.0
            for indice, linea in enumerate(lineas):
                self._check_cancel()
                wav = Path(tmp) / f"voz_{indice:02d}.wav"
                self._audio.synthesize(
                    linea.texto, self._voice, self._voice_speed, output_path=wav
                )
                duracion = self._audio_duration(wav)
                if indice < self._opening_count:
                    entries.append((cursor, wav))
                    narradas.append(cp.ChronicleLine(cursor, linea.texto))
                    cursor += duracion + cp.MIN_GAP_S
                else:
                    inicio = float(linea.t_s)
                    # No montar las voces: si la anterior aún suena, se espera; si
                    # el retraso es excesivo (el hecho ya pasó), se SALTA y se pasa
                    # al siguiente gol/parada.
                    inicio_real = max(inicio, cursor + cp.MIN_GAP_S)
                    if inicio_real - inicio > cp.MAX_NARRATION_DELAY_S:
                        saltadas += 1
                        continue
                    entries.append((inicio_real, wav))
                    narradas.append(cp.ChronicleLine(inicio_real, linea.texto))
                    cursor = inicio_real + duracion
                self.progress.emit(
                    55 + int((indice + 1) / total * 25),
                    f"Locutando la frase {indice + 1}/{total}",
                )
            if saltadas:
                logger.warning(
                    "Crónica: %d frases no se montan (se solapaban y su momento ya pasó).",
                    saltadas,
                )
            self.progress.emit(82, "Mezclando la narración…")
            comando = cp.narration_command(
                VideoAssembler._ffmpeg(), entries, destino
            )
            self._run_mix(comando, destino)
        return destino, narradas

    @staticmethod
    def _audio_duration(path: Path) -> float:
        """Duración real (segundos) de un WAV sintetizado.

        Sirve para encadenar la apertura sin solapes: la duración que se estima por
        caracteres se queda corta (los nombres y los números se locutan más lentos).
        """
        try:
            with wave.open(str(path), "rb") as manejador:
                marcos = manejador.getnframes()
                tasa = manejador.getframerate() or 1
                return marcos / float(tasa)
        except (OSError, wave.Error, ZeroDivisionError):
            logger.debug("Crónica: no se pudo medir el audio %s", path)
            return 0.0

    def _run_mix(self, argv: list[str], destino: Path) -> None:
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=MIX_TIMEOUT,
                **no_window_kwargs(),
            )
        except subprocess.TimeoutExpired as exc:
            raise EngineError(f"La mezcla de la voz no terminó en {MIX_TIMEOUT}s.") from exc
        except OSError as exc:
            raise EngineError(f"No se pudo ejecutar ffmpeg: {exc}") from exc
        if proc.returncode != 0 or not destino.is_file():
            cola = (proc.stderr or "").strip()[-600:]
            raise EngineError(f"No se pudo mezclar la narración. {cola}")

    # ---- 4. música --------------------------------------------------

    def _generate_music(self) -> Path | None:
        if not self._music_prompt:
            return None
        if self._music is None or not self._music.is_available():
            logger.info("Crónica: MusicGen no está disponible; se monta sin música.")
            return None
        self._output_dir.mkdir(parents=True, exist_ok=True)
        destino = self._output_dir / f"{cp.default_output_name(self._video)}_musica.wav"
        self.progress.emit(86, "Generando la música…")
        try:
            return self._music.generate_music(
                self._music_prompt,
                duration=self._duration,
                output_path=destino,
            )
        except EngineError as exc:
            # La música es un adorno: sin ella la crónica sigue siendo válida.
            logger.warning("Crónica: no se pudo generar la música: %s", exc)
            return None

    # ---- 5. montaje -------------------------------------------------

    def _assemble(self, lineas, narracion: Path, musica: Path | None) -> Path:
        self.progress.emit(90, "Montando el vídeo con subtítulos…")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        srt: Path | None = None
        if lineas:
            srt = self._output_dir / f"{cp.default_output_name(self._video)}.srt"
            srt.write_text(
                cp.chronicle_srt(lineas, duration=self._duration), encoding="utf-8"
            )
        try:
            final = VideoAssembler(self._output_dir).assemble(
                self._video,
                audio_path=narracion,
                srt_path=srt,
                subtitle_style=self._style,
                music_path=musica,
                music_volume=self._music_volume,
                voice_volume=self._voice_volume,
                original_volume=self._original_volume,
                ducking=True,
                keep_original=self._keep_original,
            )
        finally:
            if srt is not None:
                srt.unlink(missing_ok=True)
        entrega = final.with_name(f"{final.stem}_cronica.md")
        texto = cp.chronicle_markdown(
            lineas, video_name=self._video.name, duration=self._duration
        )
        if self._avisos:
            texto += "\n## Avisos de verificación\n" + "\n".join(
                f"- {aviso}" for aviso in self._avisos
            ) + "\n"
        entrega.write_text(texto, encoding="utf-8")
        logger.info("Crónica: %s (+ %s)", final, entrega.name)
        self.progress.emit(100, "Crónica lista.")
        return final


class ActaWorker(ThreadWorker):
    """Lee el acta del partido (PDF o foto) y devuelve el bloque «Datos del partido».

    El acta es lo único que trae nombres y dorsales; la visión sólo ve colores. Este
    worker no monta nada: deja el texto listo para la caja de la crónica, que es
    donde el usuario lo revisa antes de generar.
    """

    def __init__(
        self,
        acta_path,
        *,
        vision_engine=None,
        chat_engine=None,
        model: str | None = None,
        chat_model: str | None = None,
        scheduler=None,
        liberar_otros=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._path = Path(acta_path)
        self._vision = vision_engine
        self._chat = chat_engine
        self._model = model
        self._chat_model = chat_model if chat_model is not None else model
        self._scheduler = scheduler
        self._liberar_otros = liberar_otros

    def _check_cancel(self) -> None:
        if self.isInterruptionRequested():
            raise EngineError(CANCEL_MESSAGE)

    def work(self) -> str:
        # Leer el acta usa visión (OCR de escaneos) y chat (estructurar): una sola
        # reserva de GPU para todo, igual que la crónica.
        with gpu_slot(self._scheduler, "document", liberar_otros=self._liberar_otros):
            texto = self._read_text()
            if not texto.strip():
                raise EngineError(
                    "El acta no tiene texto legible (¿está vacía, girada o borrosa?)."
                )
            self._check_cancel()
            self.progress.emit(75, "Interpretando el acta…")
            bloque = acp.acta_context(self._interpret(texto))
            if not bloque.strip():
                raise EngineError(
                    "No se pudieron sacar equipos ni jugadores del acta. Prueba a "
                    "escribir los datos a mano en la caja."
                )
            self.progress.emit(100, "Acta leída.")
            return bloque

    # ---- lectura del archivo ---------------------------------------

    def _read_text(self) -> str:
        if self._path.suffix.lower() == ".pdf":
            return self._read_pdf()
        return self._ocr(self._path)

    def _read_pdf(self) -> str:
        if pymupdf is None:
            raise EngineError(
                "PyMuPDF no está instalado: no se puede leer el PDF del acta."
            )
        doc = pymupdf.open(str(self._path))
        try:
            total = doc.page_count
            partes: list[str] = []
            with tempfile.TemporaryDirectory(prefix="imperator_acta_") as tmp:
                for index in range(total):
                    self._check_cancel()
                    pagina = doc.load_page(index)
                    # Ordenado por posición: el acta es una tabla y así las filas
                    # salen alineadas (con el orden del flujo salían descolocadas).
                    texto = (pagina.get_text("text", sort=True) or "").strip()
                    if not texto:
                        # Escaneo sin capa de texto: se rasteriza y se hace OCR.
                        texto = self._ocr(self._render(pagina, Path(tmp), index))
                    partes.append(texto)
                    self.progress.emit(
                        int((index + 1) / max(1, total) * 60),
                        f"Acta: página {index + 1}/{total}",
                    )
            return "\n".join(partes)
        finally:
            doc.close()

    def _render(self, pagina, carpeta: Path, index: int) -> Path:
        """Página del PDF a PNG (para que la visión pueda leerla)."""
        png = carpeta / f"acta_{index:02d}.png"
        pagina.get_pixmap(dpi=ACTA_OCR_DPI).save(str(png))
        return png

    def _ocr(self, imagen: Path) -> str:
        if self._vision is None or not self._vision.is_available():
            raise EngineError(
                "No hay motor de visión (Ollama Vision) para leer el acta escaneada."
            )
        return self._vision.analyze_image(imagen, acp.ACTA_OCR_PROMPT, self._model)

    # ---- estructuración --------------------------------------------

    def _interpret(self, texto: str) -> acp.Acta:
        """Primero el lector determinista; el chat sólo si aquél no saca jugadores.

        Un acta de federación tiene una tabla reconocible, y leerla con reglas es
        exacto y no depende de ningún modelo: nada que alucinar. El chat queda para
        maquetaciones que la tabla no pille (o para el texto torcido de un OCR).
        """
        acta = acp.parse_acta(texto)
        if acta.jugadores:
            return acta
        if self._chat is not None and self._chat.is_available():
            try:
                respuesta = self._chat.chat(
                    [{"role": "user", "content": acp.ACTA_PROMPT + texto}],
                    self._chat_model,
                )
                return acp.parse_acta(respuesta)
            except EngineError as exc:
                logger.warning(
                    "Acta: el chat no pudo estructurarla (%s); se deja lo leído.", exc
                )
        return acta


class ChronicleWidget(TaskWidget):
    """Pestaña Crónica: un partido convertido en vídeo comentado por IA."""

    #: El vídeo con voz y subtítulos, listo para el Ensamblador.
    clips_ready = pyqtSignal(object)

    def __init__(self, engine_manager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._engine_manager = engine_manager
        self._prefs = prefs.PrefsStore(getattr(engine_manager, "config_manager", None))
        self._video: Path | None = None
        self._duration: float | None = None
        self._last_output: Path | None = None
        #: Carpeta del último resultado (para el botón «Abrir carpeta»).
        self._last_dir: Path | None = None
        self._pending_result = None
        #: Token que invalida las respuestas del pool en vuelo. Ver
        #: :meth:`_on_models_result`: si el widget se cierra mientras se listan
        #: los modelos, el resultado llega con un token viejo y se descarta.
        self._generation = 0
        #: Informe del Cuaderno cargado (o ``None``): permite generar sin analizar.
        self._informe_data: dict | None = None
        #: Percepción de Comprensión cargada (o ``None``): evita volver a mirar el vídeo.
        self._percepcion_data: dict | None = None
        #: Material cargado desde el que se escribe la crónica (informe o percepción).
        self._material: dict | None = None
        self._extractor = FrameExtractor(cache=FrameCache())
        self._build_ui()
        # La disciplina es compartida: se restaura y se sincroniza en vivo.
        self._restore_discipline()
        discipline.BUS.changed.connect(self._on_sport_changed)

    def closeEvent(self, event) -> None:
        # Invalida las respuestas del pool que sigan en vuelo: si llegan después
        # de cerrar, ``_on_models_result`` las descarta (C2).
        self._generation += 1
        super().closeEvent(event)

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

    # ---- UI --------------------------------------------------------

    def _build_ui(self) -> None:
        # La pestaña tiene muchos ajustes (datos, análisis, voz, música, volúmenes
        # y subtítulos): sin scroll, «Generar crónica», la barra de avance y el
        # estado quedaban por debajo del borde de la ventana, sin forma de verlos
        # ni de pulsarlos. Eso hacía pensar que la crónica «no se generaba».
        root = self._montar_contenido_scroll()

        fila = QHBoxLayout()
        self._btn_video = QPushButton("Elegir vídeo…")
        self._btn_video.clicked.connect(self._choose_video)
        self._lbl_video = QLabel("(sin vídeo)")
        fila.addWidget(self._btn_video)
        fila.addWidget(self._lbl_video, 1)
        root.addLayout(fila)

        cabecera_contexto = QHBoxLayout()
        cabecera_contexto.addWidget(
            QLabel(
                "Datos del partido (equipos, colores, alineación y marcador; así la "
                "crónica llama a los jugadores por su nombre):"
            )
        )
        cabecera_contexto.addStretch(1)
        self._btn_acta = QPushButton("Leer acta…")
        self._btn_acta.setToolTip(
            "Lee el acta del partido (PDF o foto de la federación) y rellena esta\n"
            "caja con los equipos, la alineación (dorsal y nombre) y el marcador.\n"
            "Revisa siempre lo que traiga antes de generar la crónica."
        )
        self._btn_acta.clicked.connect(self._read_acta)
        cabecera_contexto.addWidget(self._btn_acta)

        contexto = QVBoxLayout()
        contexto.addLayout(cabecera_contexto)
        self._txt_context = QPlainTextEdit()
        self._txt_context.setPlaceholderText(cp.CONTEXT_EXAMPLE)
        self._txt_context.setFixedHeight(96)
        self._txt_context.setToolTip(
            "Escribe un dato por línea. Lo que pongas aquí se toma como cierto y la\n"
            "crónica lo usa tal cual (nombres, dorsales, colores y marcador); si lo\n"
            "dejas vacío, la IA comenta sin nombres. Ejemplo:\n" + cp.CONTEXT_EXAMPLE
        )
        contexto.addWidget(self._txt_context)
        root.addLayout(contexto)

        # Instrucciones para la IA: qué quiere el usuario que cuente y con qué
        # prioridad. Van al prompt de cada fotograma y al que escribe la crónica.
        cabecera_ia = QHBoxLayout()
        cabecera_ia.addWidget(QLabel("Instrucciones para la IA (qué quieres oír):"))
        cabecera_ia.addStretch(1)
        self._btn_ejemplo_ia = QPushButton("Poner ejemplo (balonmano)")
        self._btn_ejemplo_ia.setToolTip(
            "Rellena el ejemplo recomendado: presentar la alineación al principio\n"
            "(mientras calientan) y después centrarse en goles, paradas del portero\n"
            "y lanzamientos de 7 metros. Edítalo a tu gusto."
        )
        self._btn_ejemplo_ia.clicked.connect(
            lambda: self._txt_instructions.setPlainText(sp.INSTRUCTIONS_EXAMPLE)
        )
        cabecera_ia.addWidget(self._btn_ejemplo_ia)
        root.addLayout(cabecera_ia)
        self._txt_instructions = QPlainTextEdit()
        self._txt_instructions.setFixedHeight(84)
        self._txt_instructions.setPlaceholderText(sp.INSTRUCTIONS_EXAMPLE)
        self._txt_instructions.setToolTip(
            "Escribe lo que quieres que cuente la crónica y en qué orden (p. ej. la\n"
            "alineación al principio, luego goles, paradas y 7 metros). Manda sobre\n"
            "el reparto automático de frases."
        )
        root.addWidget(self._txt_instructions)

        form = QFormLayout()

        self._cmb_language = QComboBox()
        for codigo, etiqueta in asp.LANGUAGES:
            self._cmb_language.addItem(etiqueta, codigo)
        self._set_combo_data(self._cmb_language, cp.DEFAULT_LANGUAGE)
        form.addRow("Idioma:", self._cmb_language)

        # Disciplina: sus reglas y vocabulario van a la visión y al guion. Es la
        # MISMA en Cuaderno, Crónica y Comprensión (sincronizada en vivo).
        self._cmb_sport = QComboBox()
        discipline.fill(self._cmb_sport)
        self._cmb_sport.setToolTip(
            "Disciplina del vídeo. Sus reglas y vocabulario se le pasan a la IA en\n"
            "la visión y en el guion (sin esto confunde deportes: p. ej. llama\n"
            "«voleibol» a un balonmano). Es la MISMA en Cuaderno, Crónica y\n"
            "Comprensión: cambiarla aquí la cambia en las demás."
        )
        self._cmb_sport.currentIndexChanged.connect(self._on_sport_picked)
        form.addRow("Disciplina:", self._cmb_sport)

        self._spn_lines = QSpinBox()
        self._spn_lines.setRange(3, cp.MAX_LINES)
        self._spn_lines.setValue(cp.DEFAULT_MAX_LINES)
        self._spn_lines.setToolTip(
            f"Cada frase se locuta y se subtitula (1 frase = 1 momento). Un partido\n"
            f"largo aguanta 40-60 momentos; hasta {cp.MAX_LINES}. Sube también los\n"
            "fotogramas analizados: la IA sólo puede contar lo que ha visto."
        )
        form.addRow("Frases:", self._spn_lines)

        analisis = QHBoxLayout()
        self._spn_every = QSpinBox()
        self._spn_every.setRange(1, 60)
        # Calidad por defecto: un fotograma cada 10 s (antes 30 s, y con salto tan
        # grande la visión se perdía goles y paradas: entre dos fotogramas pasaba
        # medio minuto de juego) y el tope de fotogramas repartidos por todo el vídeo.
        self._spn_every.setValue(10)
        self._spn_every.setSuffix(" s")
        self._spn_every.setToolTip(
            "Cada cuántos segundos se mira un fotograma. Con salto grande (30-60 s)\n"
            "la visión se pierde goles y paradas: para narrar jugadas, 10 s o menos.\n"
            "Con un motor de visión de pago, bajar este valor sube el coste."
        )
        self._spn_max_frames = QSpinBox()
        self._spn_max_frames.setRange(4, vu.MAX_MAX_FRAMES)
        self._spn_max_frames.setValue(vu.MAX_MAX_FRAMES)
        self._spn_max_frames.setToolTip(
            "Fotogramas que mira la visión (uno cada N segundos). Para muchas frases\n"
            "conviene subirlo: si pides 40 frases y sólo se han visto 16 momentos, el\n"
            "modelo rellena con generalidades. Para 52 min: 1 cada 60 s y 60 fotogramas."
        )
        self._chk_transcribe = QCheckBox("Transcribir el audio")
        self._chk_transcribe.setChecked(True)
        analisis.addWidget(QLabel("Un fotograma cada"))
        analisis.addWidget(self._spn_every)
        analisis.addWidget(QLabel("máx."))
        analisis.addWidget(self._spn_max_frames)
        analisis.addWidget(self._chk_transcribe)
        analisis.addStretch(1)
        form.addRow("Análisis:", analisis)

        # ---- IA de la crónica: quién mira y quién escribe -----------------
        # La crónica la firman DOS modelos: uno de visión (describe los fotogramas)
        # y otro de chat (escribe el guion). Se pueden elegir aquí para usar una IA
        # de pago o con más visión sin cambiar la del resto de la app.
        ia_vision = QHBoxLayout()
        self._cmb_vision_engine = QComboBox()
        self._cmb_vision_engine.setToolTip(
            "Motor que MIRA los fotogramas del vídeo. Ollama Vision es local y\n"
            "gratis; un motor de pago (OpenAI Vision / GPT-4o) ve mejor y describe\n"
            "con más detalle, pero cobra por fotograma. Si cargas el informe del\n"
            "Cuaderno o la percepción de Comprensión, NO se analiza el vídeo y este\n"
            "motor no se usa.\n"
            "Sólo aparecen motores que SABEN mirar: los de chat (p. ej. DeepSeek)\n"
            "no tienen visión (sólo texto) y no salen aquí."
        )
        self._cmb_vision_engine.currentIndexChanged.connect(
            self._on_vision_engine_changed
        )
        self._cmb_vision_model = QComboBox()
        self._cmb_vision_model.setToolTip(
            "Modelo de visión. La lista trae los modelos CONFIGURADOS y los que\n"
            "ofrezca el motor: los INSTALADOS en local (Ollama/LM Studio) y los de\n"
            "la API de pago (gpt-4o ve mejor; gpt-4o-mini es más barato).\n"
            "«(modelo por defecto)» usa el del motor."
        )
        self._cmb_vision_model.currentIndexChanged.connect(
            self._on_vision_model_changed
        )
        # El modelo tiene su propio selector; se le da ancho para que se lea bien el
        # nombre (antes quedaba tan estrecho que no se veía que se podía cambiar).
        self._cmb_vision_model.setMinimumContentsLength(24)
        ia_vision.addWidget(self._cmb_vision_engine, 2)
        ia_vision.addWidget(self._cmb_vision_model, 2)
        form.addRow("Visión:", ia_vision)

        ia_chat = QHBoxLayout()
        self._cmb_chat_engine = QComboBox()
        self._cmb_chat_engine.setToolTip(
            "Motor de chat que ESCRIBE la crónica. Por defecto, el elegido en la\n"
            "pestaña Chat; aquí puedes usar uno mejor sólo para la crónica\n"
            "(p. ej. OpenAI o DeepSeek).\n"
            "DeepSeek escribe texto pero NO ve imágenes: sirve para redactar la\n"
            "crónica, no para analizar los fotogramas (eso es «Visión»)."
        )
        self._cmb_chat_engine.currentIndexChanged.connect(self._on_chat_engine_changed)
        self._cmb_chat_model = QComboBox()
        self._cmb_chat_model.setToolTip(
            "Modelo que escribe la crónica. La lista trae los configurados y los que\n"
            "ofrezca el motor: locales (Ollama/LM Studio/llama.cpp) y de pago\n"
            "(OpenAI/DeepSeek…). «(modelo por defecto)» usa el del motor."
        )
        self._cmb_chat_model.currentIndexChanged.connect(self._on_chat_model_changed)
        self._cmb_chat_model.setMinimumContentsLength(24)
        ia_chat.addWidget(self._cmb_chat_engine, 2)
        ia_chat.addWidget(self._cmb_chat_model, 2)
        form.addRow("IA que escribe:", ia_chat)

        self._lbl_ia = QLabel()
        self._lbl_ia.setWordWrap(True)
        form.addRow(self._lbl_ia)

        self._cmb_motor = QComboBox()
        for etiqueta, motor in ap.MOTORS:
            self._cmb_motor.addItem(etiqueta, motor)
        self._cmb_motor.setToolTip(ap.MOTOR_TOOLTIP)
        self._cmb_motor.currentIndexChanged.connect(self._on_motor_changed)
        form.addRow("Motor de voz:", self._cmb_motor)

        self._cmb_voice = QComboBox()
        form.addRow("Voz:", self._cmb_voice)

        self._spn_voice_speed = QDoubleSpinBox()
        self._spn_voice_speed.setRange(ap.SPEED_MIN, ap.SPEED_MAX)
        self._spn_voice_speed.setSingleStep(ap.SPEED_STEP)
        self._spn_voice_speed.setValue(ap.SPEED_DEFAULT)
        self._spn_voice_speed.setSuffix(" ×")
        form.addRow("Velocidad de la voz:", self._spn_voice_speed)

        musica = QHBoxLayout()
        self._chk_music = QCheckBox("Música de fondo (MusicGen)")
        self._txt_music = QLineEdit()
        self._txt_music.setPlaceholderText("p. ej. épico, percusión, pabellón")
        self._spn_music_volume = QDoubleSpinBox()
        self._spn_music_volume.setRange(0.0, 1.0)
        self._spn_music_volume.setSingleStep(0.05)
        self._spn_music_volume.setValue(0.25)
        musica.addWidget(self._chk_music)
        musica.addWidget(self._txt_music, 1)
        musica.addWidget(QLabel("Vol.:"))
        musica.addWidget(self._spn_music_volume)
        form.addRow("Música:", musica)

        self._chk_original = QCheckBox("Conservar el audio original bajo la voz")
        self._chk_original.setChecked(True)
        self._chk_original.setToolTip(
            "Deja el sonido de la grabación (público, silbato, balón) y pone la voz\n"
            "encima. Desmárcalo para oír solo la narración. No añade música: eso es\n"
            "otra casilla, y está apagada por defecto."
        )
        form.addRow("Audio del vídeo:", self._chk_original)

        volumenes = QHBoxLayout()
        self._spn_original_volume = QSpinBox()
        self._spn_original_volume.setRange(vep.VOLUME_MIN, vep.VOLUME_MAX)
        self._spn_original_volume.setValue(int(cp.DEFAULT_ORIGINAL_VOLUME * 100))
        self._spn_original_volume.setSuffix(" %")
        self._spn_original_volume.setToolTip(
            "Volumen del sonido de la grabación (pabellón, silbato) bajo la voz.\n"
            "Bájalo si el ambiente tapa al locutor; súbelo si casi no se oye."
        )
        self._spn_voice_volume = QSpinBox()
        self._spn_voice_volume.setRange(vep.VOLUME_MIN, vep.VOLUME_MAX)
        self._spn_voice_volume.setValue(100)
        self._spn_voice_volume.setSuffix(" %")
        self._spn_voice_volume.setToolTip(
            "Volumen de la narración (la voz del locutor)."
        )
        volumenes.addWidget(QLabel("Vol. vídeo:"))
        volumenes.addWidget(self._spn_original_volume)
        volumenes.addWidget(QLabel("Vol. voz:"))
        volumenes.addWidget(self._spn_voice_volume)
        volumenes.addStretch(1)
        form.addRow("Volúmenes:", volumenes)

        subtitulos = QHBoxLayout()
        self._spn_fontsize = QSpinBox()
        self._spn_fontsize.setRange(12, 96)
        self._spn_fontsize.setValue(asp.DEFAULT_FONTSIZE)
        self._cmb_position = QComboBox()
        for clave, etiqueta in asp.POSITIONS:
            self._cmb_position.addItem(etiqueta, clave)
        self._cmb_color = QComboBox()
        for clave, etiqueta in asp.COLORS:
            self._cmb_color.addItem(etiqueta, clave)
        subtitulos.addWidget(QLabel("Tamaño:"))
        subtitulos.addWidget(self._spn_fontsize)
        subtitulos.addWidget(QLabel("Posición:"))
        subtitulos.addWidget(self._cmb_position)
        subtitulos.addWidget(QLabel("Color:"))
        subtitulos.addWidget(self._cmb_color)
        subtitulos.addStretch(1)
        form.addRow("Subtítulos:", subtitulos)

        root.addLayout(form)

        # Material ya analizado (Comprensión y Cuaderno «echan una mano»): con él la
        # crónica se escribe sin volver a mirar el vídeo (ni visión ni Whisper).
        fila_informe = QHBoxLayout()
        self._btn_informe = QPushButton("Usar informe del Cuaderno…")
        self._btn_informe.setToolTip(
            "Carga el informe.json que deja la pestaña Cuaderno (descripciones por\n"
            "fotograma + audio). Con él, la crónica se escribe SIN volver a analizar\n"
            "el vídeo (ni visión ni Whisper): reutiliza lo que el Cuaderno ya vio."
        )
        self._btn_informe.clicked.connect(self._load_informe_dialog)
        self._btn_percepcion = QPushButton("Usar percepción de Comprensión…")
        self._btn_percepcion.setToolTip(
            "Carga el comprension.json que deja la pestaña Comprensión (anotación por\n"
            "fotograma con objetos/etiquetas/OCR, transcripción y resumen). Con él,\n"
            "la crónica se escribe SIN volver a mirar el vídeo: es la vía más rápida\n"
            "y no gasta visión (y, con una IA de pago, no cuesta por fotograma)."
        )
        self._btn_percepcion.clicked.connect(self._load_percepcion_dialog)
        self._chk_informe = QCheckBox("Usar el material cargado (no volver a analizar)")
        self._chk_informe.setEnabled(False)
        self._chk_informe.toggled.connect(self._on_material_toggled)
        fila_informe.addWidget(self._btn_informe)
        fila_informe.addWidget(self._btn_percepcion)
        fila_informe.addWidget(self._chk_informe)
        fila_informe.addStretch(1)
        root.addLayout(fila_informe)

        self._lbl_flujo = QLabel(
            "Ayuda para una crónica exacta: Comprensión analiza el vídeo una vez\n"
            "(comprension.json) → el Cuaderno lo convierte en informe.json → aquí lo\n"
            "cargas y generas. Sin material, la IA mira el vídeo con el motor de visión\n"
            "elegido arriba."
        )
        self._lbl_flujo.setWordWrap(True)
        root.addWidget(self._lbl_flujo)

        acciones = QHBoxLayout()
        self._btn_generate = QPushButton("Generar crónica")
        self._btn_generate.clicked.connect(self._generate)
        self._btn_send = QPushButton("Enviar al Ensamblador")
        self._btn_send.setEnabled(False)
        self._btn_send.clicked.connect(self._send_to_assembler)
        self._btn_cancel = QPushButton("Cancelar")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)
        self._btn_open_out = QPushButton("Abrir carpeta del resultado")
        self._btn_open_out.setEnabled(False)
        self._btn_open_out.clicked.connect(self._open_results_folder)
        acciones.addWidget(self._btn_generate)
        acciones.addWidget(self._btn_send)
        acciones.addWidget(self._btn_open_out)
        acciones.addWidget(self._btn_cancel)
        acciones.addStretch(1)
        root.addLayout(acciones)

        self._montar_progreso(root)
        root.addStretch(1)

        self._on_motor_changed()
        self._populate_ia()
        self._restore_instructions()

    def _set_combo_data(self, combo: QComboBox, data) -> None:
        index = combo.findData(data)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _on_motor_changed(self) -> None:
        """Rellena las voces del motor elegido (listarlas es local y barato)."""
        engine = self._engine_for(self._cmb_motor.currentData())
        self._cmb_voice.clear()
        if engine is None:
            self._set_status("Ese motor de voz no está disponible.")
            return
        try:
            voces = list(engine.list_voices())
        except Exception as exc:  # noqa: BLE001 - listar voces no debe romper la UI
            logger.debug("Crónica: no se pudieron listar las voces: %r", exc)
            voces = []
        motor = self._cmb_motor.currentData()
        for voz in ap.sort_voices(voces):
            self._cmb_voice.addItem(ap.voice_label(voz, motor), voz)
        elegida = ap.resolve_voice(
            self._cmb_motor.currentData(), None, voces
        )
        if elegida is not None and elegida in voces:
            self._cmb_voice.setCurrentIndex(voces.index(elegida))

    def _engine_for(self, motor: str | None):
        ids = {
            "piper": "piper_tts",
            "kokoro": "kokoro_tts",
            "voice_clone": "voice_clone",
        }
        engine_id = ids.get(motor or "")
        return self._engine_manager.get_engine(engine_id) if engine_id else None

    # ---- IA de la crónica (visión + chat) ---------------------------

    def _vision_engine_ids(self) -> list[str]:
        """Ids de motores que saben mirar una imagen (``analyze_image``)."""
        ids: list[str] = []
        for config in self._engine_manager.list_configs():
            engine_id = str(config.get("id") or "")
            if not engine_id:
                continue
            instancia = self._engine_manager.get_engine(engine_id)
            if instancia is None:
                continue
            if callable(getattr(instancia, "analyze_image", None)):
                ids.append(engine_id)
        return ids

    def _chat_engine_ids(self) -> list[str]:
        """Ids de motores de chat que saben escribir (``chat``)."""
        ids: list[str] = []
        for config in self._engine_manager.list_configs():
            engine_id = str(config.get("id") or "")
            if not engine_id:
                continue
            instancia = self._engine_manager.get_engine(engine_id)
            if instancia is None:
                continue
            if callable(getattr(instancia, "chat", None)):
                ids.append(engine_id)
        return ids

    @staticmethod
    def _nombre_motor(config: dict, engine_id: str) -> str:
        return str(config.get("nombre") or engine_id)

    def _fill_engines(self, combo: QComboBox, ids: list[str], pref_key: str, prefer) -> None:
        """Rellena un selector de motores y restaura el guardado (sin reemitir)."""
        combo.blockSignals(True)
        combo.clear()
        for engine_id in ids:
            config = self._engine_manager.get_config(engine_id) or {}
            combo.addItem(self._nombre_motor(config, engine_id), engine_id)
        if not ids:
            combo.addItem("(ninguno disponible)", None)
        else:
            guardado = self._prefs.get_str(pref_key)
            objetivo = guardado if guardado in ids else (prefer if prefer in ids else None)
            if objetivo:
                indice = combo.findData(objetivo)
                if indice >= 0:
                    combo.setCurrentIndex(indice)
        combo.blockSignals(False)

    def _fill_models(
        self,
        combo: QComboBox,
        engine_combo: QComboBox,
        engine_id: str | None,
        pref_key: str,
    ) -> None:
        """Modelos del motor elegido: primero los configurados (sin red) y luego los
        **reales** que ofrezca el motor (Ollama local, LM Studio, nube…), en segundo
        plano para no bloquear la UI.

        Así se puede elegir tanto un modelo **local** (los instalados en Ollama o
        servidos por LM Studio/llama.cpp) como uno **de pago** (los que exponga la
        API), sin tocar ``engines.json``.
        """
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("(modelo por defecto)", None)
        params = {}
        if engine_id:
            config = self._engine_manager.get_config(engine_id) or {}
            params = config.get("parametros") or {}
        nombres = [m for m in params.get("models") or [] if isinstance(m, str) and m]
        por_defecto = params.get("model")
        if isinstance(por_defecto, str) and por_defecto and por_defecto not in nombres:
            nombres.insert(0, por_defecto)
        for nombre in nombres:
            combo.addItem(nombre, nombre)
        guardado = self._prefs.get_str(pref_key)
        if guardado:
            indice = combo.findData(guardado)
            if indice >= 0:
                combo.setCurrentIndex(indice)
        combo.setEnabled(bool(engine_id))
        combo.blockSignals(False)
        self._refresh_models(combo, engine_combo, engine_id)

    def _refresh_models(
        self, combo: QComboBox, engine_combo: QComboBox, engine_id: str | None
    ) -> None:
        """Añade en segundo plano los modelos que ofrece el motor (instalados o API)."""
        if not engine_id:
            return
        engine = self._engine_manager.get_engine(engine_id)
        listar = getattr(engine, "list_models", None)
        if not callable(listar):
            return
        token = self._generation
        run_pool_task(
            listar,
            on_result=lambda modelos, t=token: self._on_models_result(
                t, combo, engine_combo, engine_id, modelos
            ),
            on_error=lambda mensaje: logger.debug(
                "Crónica: no se pudieron listar los modelos de %s: %s",
                engine_id,
                mensaje,
            ),
        )

    def _on_models_result(
        self,
        token: int,
        combo: QComboBox,
        engine_combo: QComboBox,
        engine_id: str,
        modelos,
    ) -> None:
        """Aplica el listado de modelos si el widget y los combos siguen vivos."""
        # C2: el callback del pool puede llegar después de cerrar la pestaña. Los
        # QComboBox ya no existen y tocarlos abortaba el proceso con "wrapped
        # C/C++ object of type QComboBox has been deleted".
        if sip.isdeleted(self) or token != self._generation:
            return
        if sip.isdeleted(combo) or sip.isdeleted(engine_combo):
            return
        self._merge_models(combo, engine_combo, engine_id, modelos)

    def _merge_models(
        self,
        combo: QComboBox,
        engine_combo: QComboBox,
        engine_id: str,
        modelos,
    ) -> None:
        """Fusiona los modelos devueltos por el motor sin pisar lo que eligió el usuario."""
        if engine_combo.currentData() != engine_id:
            return  # el usuario ya cambió de motor: este resultado es viejo
        elegido = combo.currentData()
        existentes = {combo.itemData(i) for i in range(combo.count())}
        combo.blockSignals(True)
        for nombre in modelos or []:
            nombre = str(nombre)
            if nombre and nombre not in existentes:
                combo.addItem(nombre, nombre)
                existentes.add(nombre)
        combo.blockSignals(False)
        if elegido:
            indice = combo.findData(elegido)
            if indice >= 0:
                combo.setCurrentIndex(indice)

    def _populate_ia(self) -> None:
        """Rellena los selectores de IA (visión y chat) y sus modelos."""
        self._fill_engines(
            self._cmb_vision_engine,
            self._vision_engine_ids(),
            prefs.CHRONICLE_VISION_ENGINE,
            prefer="ollama_vision",
        )
        self._on_vision_engine_changed()
        self._fill_engines(
            self._cmb_chat_engine,
            self._chat_engine_ids(),
            prefs.CHRONICLE_CHAT_ENGINE,
            prefer=ce.preferred_chat_engine_id(
                getattr(self._engine_manager, "config_manager", None)
            ),
        )
        self._on_chat_engine_changed()

    def _on_vision_engine_changed(self) -> None:
        engine_id = self._cmb_vision_engine.currentData()
        self._fill_models(
            self._cmb_vision_model,
            self._cmb_vision_engine,
            engine_id,
            prefs.CHRONICLE_VISION_MODEL,
        )
        self._prefs.set(prefs.CHRONICLE_VISION_ENGINE, engine_id or "")
        self._update_ia_hint()

    def _on_vision_model_changed(self) -> None:
        self._prefs.set(
            prefs.CHRONICLE_VISION_MODEL, self._cmb_vision_model.currentData() or ""
        )

    def _on_chat_engine_changed(self) -> None:
        engine_id = self._cmb_chat_engine.currentData()
        self._fill_models(
            self._cmb_chat_model,
            self._cmb_chat_engine,
            engine_id,
            prefs.CHRONICLE_CHAT_MODEL,
        )
        self._prefs.set(prefs.CHRONICLE_CHAT_ENGINE, engine_id or "")
        self._update_ia_hint()

    def _on_chat_model_changed(self) -> None:
        self._prefs.set(
            prefs.CHRONICLE_CHAT_MODEL, self._cmb_chat_model.currentData() or ""
        )

    @staticmethod
    def _es_vision_local(engine_id: str | None) -> bool:
        """¿El motor de visión es local (Ollama) y, por tanto, sin coste de API?"""
        return not engine_id or engine_id == "ollama_vision"

    def _hay_material_cargado(self) -> bool:
        """¿Se va a escribir desde material ya analizado (sin volver a mirar)?"""
        return self._material is not None and self._chk_informe.isChecked()

    def _update_ia_hint(self) -> None:
        """Explica qué IA se usará y avisa del gasto si es de pago."""
        usar_material = self._hay_material_cargado()
        if usar_material:
            origen = "la percepción de Comprensión" if self._percepcion_data else "el informe del Cuaderno"
            self._lbl_ia.setText(
                f"Se usará {origen} cargado: NO se analizará el vídeo y el motor de "
                "visión no se usa (así no se gasta ni GPU ni tokens de visión). "
                "Desmarca la casilla para volver a mirar el vídeo."
            )
        elif not self._es_vision_local(self._cmb_vision_engine.currentData()):
            self._lbl_ia.setText(
                "IA de pago: cada fotograma analizado consume tokens y cuesta dinero. "
                "Cuantos más fotogramas («Un fotograma cada… · máx.») y más frases, "
                "mayor el gasto. Reutilizar el informe del Cuaderno o la percepción "
                "de Comprensión analiza el vídeo sin coste de API."
            )
        else:
            self._lbl_ia.setText(
                "Con Ollama Vision el análisis es local y gratis. Para una crónica "
                "más fina puedes elegir un motor de visión de pago (OpenAI Vision). "
                "Comprensión y Cuaderno dejan su material para reutilizarlo aquí."
            )
        self._cmb_vision_engine.setEnabled(not usar_material)
        self._cmb_vision_model.setEnabled(not usar_material and bool(self._cmb_vision_engine.currentData()))

    def _on_material_toggled(self, _checked: bool) -> None:
        self._update_ia_hint()

    def _restore_instructions(self) -> None:
        """Recupera las instrucciones guardadas (o pone el ejemplo de balonmano)."""
        guardado = self._prefs.get_str(prefs.CHRONICLE_INSTRUCTIONS)
        self._txt_instructions.setPlainText(
            guardado if (guardado or "").strip() else sp.INSTRUCTIONS_EXAMPLE
        )

    def _vision_engine(self):
        elegido = self._cmb_vision_engine.currentData() or "ollama_vision"
        return self._engine_manager.get_engine(elegido)

    def _chat_engine(self):
        """Motor de chat elegido aquí; si no hay, el de la pestaña Chat."""
        elegido = self._cmb_chat_engine.currentData()
        if elegido:
            instancia = self._engine_manager.get_engine(elegido)
            if instancia is not None:
                return instancia
        return ce.preferred_chat_engine(self._engine_manager)

    def _material_report(self) -> str:
        """Informe con el que escribir la crónica (o cadena vacía para analizar)."""
        if not self._chk_informe.isChecked():
            return ""
        if self._informe_data is not None:
            return cp.informe_report(self._informe_data)
        if self._percepcion_data is not None:
            return cp.perception_report(self._percepcion_data)
        return ""

    def _material_context(self) -> str:
        """Datos del partido del material cargado (para heredar la alineación)."""
        for data in (self._informe_data, self._percepcion_data):
            if data is not None:
                contexto = str(data.get("contexto") or "")
                if contexto.strip():
                    return contexto
        return ""

    def _set_material(self, data: dict | None, es_percepcion: bool) -> None:
        """Fija el material activo: informe del Cuaderno o percepción de Comprensión."""
        if es_percepcion:
            self._percepcion_data = data
            self._informe_data = None
        else:
            self._informe_data = data
            self._percepcion_data = None
        self._material = data
        self._chk_informe.setEnabled(data is not None)
        self._chk_informe.setChecked(data is not None)
        self._update_ia_hint()

    def _busy_widgets(self) -> tuple:
        return (
            self._btn_video, self._btn_acta, self._txt_context,
            self._btn_generate, self._cmb_language, self._cmb_sport, self._spn_lines,
            self._spn_every, self._spn_max_frames, self._chk_transcribe,
            self._cmb_vision_engine, self._cmb_vision_model,
            self._cmb_chat_engine, self._cmb_chat_model,
            self._cmb_motor, self._cmb_voice, self._spn_voice_speed,
            self._chk_music, self._txt_music, self._spn_music_volume,
            self._chk_original, self._spn_original_volume, self._spn_voice_volume,
            self._spn_fontsize, self._cmb_position,
            self._cmb_color, self._btn_send,
            self._btn_informe, self._btn_percepcion, self._chk_informe,
            self._txt_instructions, self._btn_ejemplo_ia,
        )

    def _busy_only_widgets(self) -> tuple:
        return (self._btn_cancel,)

    def _on_busy_changed(self, busy: bool) -> None:
        if not busy:
            self._btn_send.setEnabled(self._last_output is not None)

    # ---- acciones ---------------------------------------------------

    def _read_acta(self) -> None:
        """Lee el acta elegida y vuelca los datos en «Datos del partido»."""
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Elegir acta del partido", "", acp.ACTA_FILTER
        )
        if not ruta:
            return
        worker = ActaWorker(
            ruta,
            vision_engine=self._vision_engine(),
            chat_engine=self._chat_engine(),
            model=self._cmb_vision_model.currentData(),
            chat_model=self._cmb_chat_model.currentData(),
            scheduler=self._gpu_scheduler(),
            liberar_otros=self._gpu_liberar_otros,
        )
        self._start(worker, on_result=self._on_acta)

    def _on_acta(self, bloque) -> None:
        self._txt_context.setPlainText(str(bloque))
        self._set_status("Acta leída. Revisa «Datos del partido» antes de generar.")

    def _choose_video(self) -> None:
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Elegir vídeo", "", vu.VIDEO_FILTER_ALL
        )
        if not ruta:
            return
        self._video = Path(ruta)
        self._lbl_video.setText(self._video.name)
        datos = {}
        try:
            datos = VideoAssembler().probe(self._video)
        except EngineError as exc:
            logger.debug("Crónica: no se pudo sondear el vídeo: %s", exc)
        self._duration = datos.get("duration")
        if self._duration:
            self._lbl_video.setText(
                f"{self._video.name} · {vu.format_timestamp(self._duration)}"
            )

    def _subtitle_style(self) -> dict:
        return asp.build_subtitle_style(
            self._spn_fontsize.value(),
            self._cmb_position.currentData(),
            self._cmb_color.currentData(),
        )

    def _load_informe_dialog(self) -> None:
        """Elige el ``informe.json`` que dejó el Cuaderno y lo carga."""
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Informe del Cuaderno", str(NOTES_DIR), "Informe (*.json);;Todos (*)"
        )
        if ruta:
            self._load_informe(ruta)

    def _load_informe(self, ruta) -> None:
        """Carga un informe del Cuaderno y deja la crónica lista para usarlo."""
        try:
            data = nts.load_informe(ruta)
        except nts.NotesError as exc:
            self._set_status(str(exc))
            return
        self._set_material(data, es_percepcion=False)
        # Si la caja «Datos del partido» está vacía, se rellena con lo que traía el
        # informe (equipos, alineación, colores, marcador): así el usuario la ve y
        # la crónica usa los mismos nombres que el Cuaderno.
        contexto_informe = str(data.get("contexto") or "").strip()
        if contexto_informe and not self._txt_context.toPlainText().strip():
            self._txt_context.setPlainText(contexto_informe)
        # Si el informe recuerda el vídeo y sigue ahí, se carga solo.
        video = str(data.get("video_path") or "")
        if video and Path(video).is_file():
            self._video = Path(video)
            self._lbl_video.setText(self._video.name)
            try:
                self._duration = VideoAssembler().probe(self._video).get("duration")
            except EngineError:
                self._duration = data.get("duracion_s")
        fotogramas = len(nts.informe_frames(data))
        aviso = ""
        if not contexto_informe:
            # Informes antiguos (o análisis sin contexto): no traen la alineación,
            # así que la crónica no podrá poner nombres hasta que se escriba aquí o
            # se vuelva a analizar en el Cuaderno con los datos del partido.
            aviso = (
                " Ojo: no trae «Datos del partido» (alineación); "
                "escríbela abajo si quieres nombres."
            )
        self._set_status(
            f"Informe del Cuaderno cargado ({fotogramas} fotogramas). "
            f"Se usará al generar la crónica.{aviso}"
        )

    def _load_percepcion_dialog(self) -> None:
        """Elige el ``comprension.json`` que dejó Comprensión y lo carga."""
        ruta, _ = QFileDialog.getOpenFileName(
            self,
            "Percepción de Comprensión",
            str(PERCEPTION_DIR),
            "Percepción (*.json);;Todos (*)",
        )
        if ruta:
            self._load_percepcion(ruta)

    def _load_percepcion(self, ruta) -> None:
        """Carga la percepción de Comprensión: la crónica se escribe sin mirar el vídeo."""
        try:
            data = vu.load_understanding(ruta)
        except vu.UnderstandingError as exc:
            self._set_status(str(exc))
            return
        self._set_material(data, es_percepcion=True)
        contexto = str(data.get("contexto") or "").strip()
        if contexto and not self._txt_context.toPlainText().strip():
            self._txt_context.setPlainText(contexto)
        video = str(data.get("video_path") or "")
        if video and Path(video).is_file():
            self._video = Path(video)
            self._lbl_video.setText(self._video.name)
            try:
                self._duration = VideoAssembler().probe(self._video).get("duration")
            except EngineError:
                self._duration = data.get("duracion_s")
        anotaciones = len(vu.understanding_frames(data))
        aviso = "" if contexto else (
            " Ojo: no trae «Datos del partido» (alineación); escríbela abajo si "
            "quieres nombres."
        )
        self._set_status(
            f"Percepción de Comprensión cargada ({anotaciones} anotaciones). "
            f"Se usará al generar la crónica sin volver a mirar el vídeo.{aviso}"
        )

    def _generate(self) -> None:
        if self._video is None:
            self._set_status("Elige primero un vídeo.")
            return
        informe = self._material_report()
        # Los datos del partido (equipos, alineación, colores) pueden venir de esta
        # caja o del material cargado (informe del Cuaderno / percepción de
        # Comprensión). Si aquí está vacía y el material los trae, se usan los del
        # material: así la crónica llama a las jugadoras por su nombre.
        context = self._txt_context.toPlainText()
        if not context.strip() and self._chk_informe.isChecked():
            context = self._material_context()
        instructions = self._txt_instructions.toPlainText()
        self._prefs.set(prefs.CHRONICLE_INSTRUCTIONS, instructions)
        worker = ChronicleWorker(
            self._video,
            extractor=self._extractor,
            vision_engine=self._vision_engine(),
            chat_engine=self._chat_engine(),
            whisper_engine=self._engine_manager.get_engine("whisper"),
            audio_engine=self._engine_for(self._cmb_motor.currentData()),
            music_engine=self._engine_manager.get_engine("musicgen"),
            output_dir=OUTPUT,
            model=None,
            vision_model=self._cmb_vision_model.currentData(),
            chat_model=self._cmb_chat_model.currentData(),
            voice=self._cmb_voice.currentData(),
            voice_speed=self._spn_voice_speed.value(),
            language=self._cmb_language.currentData(),
            max_lines=self._spn_lines.value(),
            every_s=float(self._spn_every.value()),
            max_frames=self._spn_max_frames.value(),
            transcribe=self._chk_transcribe.isChecked(),
            music_prompt=self._txt_music.text() if self._chk_music.isChecked() else "",
            music_volume=self._spn_music_volume.value(),
            voice_volume=self._spn_voice_volume.value() / 100.0,
            original_volume=self._spn_original_volume.value() / 100.0,
            keep_original=self._chk_original.isChecked(),
            subtitle_style=self._subtitle_style(),
            context=context,
            sport=self._cmb_sport.currentData(),
            instructions=instructions,
            report=informe,
            scheduler=self._gpu_scheduler(),
            liberar_otros=self._gpu_liberar_otros,
        )
        self._start(worker)

    def _start(self, worker: ThreadWorker, on_result=None) -> None:
        self._worker = worker
        self._pending_result = on_result
        worker.progress.connect(self._on_progress)
        worker.result.connect(self._on_result)
        worker.failed.connect(self._on_error)
        worker.finished.connect(self._on_finished)
        self._empezar_progreso()
        self._set_busy(True)
        self._set_status("Trabajando…")
        worker.start()

    def _on_result(self, path) -> None:
        pendiente = self._pending_result
        self._pending_result = None
        if pendiente is not None:
            # Trabajos que no producen un vídeo (p. ej. leer el acta): su resultado
            # es texto y no debe tocar el «último montaje» ni el botón de enviar.
            pendiente(path)
            return
        self._last_output = Path(path) if path else None
        self._btn_send.setEnabled(self._last_output is not None)
        if self._last_output is not None:
            self._last_dir = self._last_output.parent
            self._btn_open_out.setEnabled(True)
        self._set_status(f"Crónica lista: {path}")

    def _open_results_folder(self) -> None:
        """Abre en el Explorador la carpeta donde está la crónica montada."""
        destino = self._last_dir or OUTPUT
        if not Path(destino).is_dir():
            self._set_status(f"La carpeta todavía no existe: {destino}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(destino)))
        self._set_status(f"Carpeta abierta: {destino}")

    def _send_to_assembler(self) -> None:
        if self._last_output is None:
            self._set_status("Todavía no hay ninguna crónica montada.")
            return
        self.clips_ready.emit(str(self._last_output))
        self._set_status("Crónica enviada al Ensamblador.")

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.requestInterruption()
