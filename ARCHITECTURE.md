# Arquitectura — Imperator

Documento de referencia de la arquitectura real de **Imperator**: plataforma
local de IA para crear vídeos y TikToks, con backend Python y GUI PyQt6.
Describe cómo está construido el sistema hoy, cómo fluyen los datos y cómo
extenderlo.

## 1. Visión general

Imperator unifica en una única app de escritorio los motores de IA locales
(Ollama, ComfyUI, Piper, Kokoro, faster-whisper, MusicGen y, opcionalmente,
YOLO/CLIP/PaddleOCR para percepción estructurada) y las tareas de alto nivel que
el usuario ve como pestañas: Chat, Documentos, Imagen, Vídeo, **Comprensión**,
Audio y Ensamblador.

### Hardware objetivo

- GPU: NVIDIA RTX 5080 (16 GB VRAM, `sm_120`).
- RAM: 32 GB.

### Stack tecnológico

| Capa | Tecnología |
| ------ | ------------ |
| UI | PyQt6 (señales/slots, `QThread`, `QTabWidget`) |
| Lenguaje | Python 3.12 |
| Motores | Ollama (HTTP), ComfyUI (HTTP + WebSocket), Piper TTS (ONNX), Kokoro TTS (PyTorch CPU), faster-whisper (CTranslate2), MusicGen (audiocraft, opcional) |
| Utilidades | `requests`, `websocket-client`, `imageio-ffmpeg`, `PyMuPDF`, `python-dotenv` |

### Principios de diseño

- **Plugins, no núcleo monolítico.** Añadir un motor o una tarea no exige
  tocar el núcleo: se registra en un mapa de fábrica.
- **Configuración en `config/*.json`.** Motores, APIs y ajustes en archivos
  separados, validados por esquema e ignorados por git (solo se versionan las
  plantillas `*.example.json`).
- **Secretos fuera del código.** Las claves de API se leen de `.env`.
- **UI desacoplada de la inferencia.** La UI orquesta `QThread`s; los motores
  son bloqueantes y no conocen Qt.

## 2. Estructura de carpetas

```
C:\Imperator\
├── main.py                 # Punto de entrada: QApplication + MainWindow
├── ARCHITECTURE.md         # Este documento
├── PLAN.md                 # Histórico de decisiones (parcialmente obsoleto)
├── README.md
├── requirements.txt        # Runtime (incluye los motores locales pesados)
├── requirements-test.txt   # Mínimo para tests/CI (sin motores pesados)
├── pyrightconfig.json      # Config del analizador (extraPaths + venv)
├── .gitignore
├── core/                   # Núcleo (sin Qt, salvo task_manager)
│   ├── config_manager.py   # Lectura/escritura de config/*.json + validación
│   ├── engine_manager.py   # Registro y ciclo de vida de motores (ENGINE_FACTORY)
│   ├── api_manager.py      # Registro de APIs externas + claves desde .env
│   ├── task_manager.py     # Registro de tareas (TASK_FACTORY) + orquestador
│   ├── gpu_scheduler.py    # Cola con prioridad + sondeo de VRAM (nvidia-smi)
│   ├── frame_extractor.py  # Fotogramas y audio de un vídeo (ffmpeg)
│   ├── paths.py            # Rutas compartidas (PROJECT_ROOT, ASSETS)
│   └── video_assembler.py  # Ensamblado de vídeo con ffmpeg (no es un Engine)
├── engines/                # Un motor por fichero
│   ├── base_engine.py      # Contrato Engine + EngineError
│   ├── ollama.py           # Chat LLM
│   ├── ollama_vision.py    # Visión (OCR/resumen/tablas)
│   ├── comfyui.py          # Generación de imagen (Flux / Kontext)
│   ├── bfl.py              # Generación de imagen (Black Forest Labs, API de pago)
│   ├── wanvideo.py         # Generación de vídeo (Wan 2.2)
│   ├── piper_tts.py        # TTS rápido (ONNX)
│   ├── kokoro_tts.py       # TTS de calidad (CPU)
│   ├── whisper_engine.py   # Transcripción a SRT
│   ├── musicgen.py         # Música a partir de texto (opcional: audiocraft)
│   └── perception.py       # YOLO (detección), CLIP (etiquetas) y OCR (opcional)
├── tasks/                  # Una tarea por fichero
│   ├── base_task.py        # Contrato Task
│   ├── chat.py             # Chat
│   ├── document.py         # Documentos
│   ├── image.py            # Imagen
│   ├── video.py            # Vídeo
│   ├── understanding.py    # Comprensión de vídeo (fotogramas + audio + resumen)
│   ├── audio.py            # Audio (Piper + Kokoro)
│   └── assembler.py        # Ensamblador
├── ui/                     # Un widget por tarea
│   ├── main_window.py      # Ventana principal (QTabWidget + menú de motores)
│   ├── engines_dialog.py   # Gestión CRUD de motores
│   ├── task_widget.py      # Base común: estado y ciclo de vida del worker
│   ├── workers.py          # ThreadWorker / PoolTask (QThread, QThreadPool)
│   ├── markdown.py         # Markdown del chat (tema claro/oscuro)
│   ├── chat_widget.py
│   ├── document_widget.py
│   ├── image_widget.py
│   ├── video_widget.py
│   ├── understanding_widget.py
│   ├── audio_widget.py
│   └── assembler_widget.py
├── services/               # Lógica pura sin Qt, extraída de los widgets
│   ├── assembler_params.py # Presets y estilos del Ensamblador
│   ├── audio_params.py     # Motores TTS, rango de velocidad y voces
│   ├── chat_params.py      # Ventana de historial que se envía al motor
│   ├── document_params.py  # Formatos, prompts y helpers de documentos
│   ├── document_pipeline.py# Plan por fases del procesado de documentos
│   ├── image_params.py     # Presets y parámetros de imagen
│   ├── preferences.py      # Preferencias tipadas (PrefsStore + dataclasses)
│   ├── intent_router.py    # Router de intención del chat (a qué tarea apunta)
│   ├── output_cleanup.py   # Retención de output/ (qué se borra y por qué)
│   ├── task_orchestrator.py# Resuelve la intención contra las tareas registradas
│   ├── system_stats.py     # Sondeo de /system_stats de ComfyUI
│   ├── video_params.py     # Presets y parámetros de vídeo
│   └── video_understanding.py # Muestreo de fotogramas, prompts e informe
├── config/                 # Config real (gitignored) + plantillas (versionadas)
│   ├── engines.json        # Motores (real)
│   ├── apis.json           # APIs externas (aún sin usar)
│   ├── settings.json       # settings + last_models + unload_on_tab_change
│   ├── engines.example.json
│   ├── apis.example.json
│   └── settings.example.json
├── voices/                 # Voces ONNX de Piper (gitignored)
├── music/                  # Música de fondo para el ensamblador (gitignored)
├── output/                 # Imágenes, vídeos y audios generados (gitignored)
│                           #   Con retención al arrancar: services/output_cleanup.py
├── tools/                  # Scripts auxiliares
│   ├── generar_narracion_basajaun.py  # Narración con Piper
│   └── make_icon.py        # Regenera el icono (requiere Pillow; ver uso)
├── tests/                  # Tests (stdlib unittest; sin deps de runtime)
├── .github/workflows/      # CI: compileall + unittest
└── assets/                 # Icono, splash y logo (versionado)
```

## 3. Arquitectura de motores

### Interfaces de motor (familias)

`engines/base_engine.py` define un ABC base y una familia por tipo de motor, de
modo que cada motor declare solo lo que sabe hacer (un motor de imagen ya no
finge implementar `chat()`):

```python
class Engine(ABC):          # común: id, nombre, is_available(), list_models()
class ChatEngine(Engine):   # chat(messages, model=None) -> str
class VisionEngine(Engine): # analyze_image(image_path, prompt, model=None) -> str
class ImageEngine(Engine):  # generate_image(prompt, **kwargs) -> Path
class VideoEngine(Engine):  # generate_video(prompt, **kwargs) -> Path
class AudioEngine(Engine):  # synthesize(text, voice=None, **kwargs) -> Path
class MusicEngine(Engine):  # generate_music(prompt, **kwargs) -> Path
class TranscriptionEngine(Engine):  # transcribe(audio_path, **kwargs) -> Path
class PerceptionEngine(Engine):     # perceive(image_path, prompt="") -> Perception
```

`PerceptionEngine` devuelve un `Perception` **estructurado** (objetos, etiquetas y
texto leído) en vez de lenguaje natural: es lo que aportan YOLO, CLIP y el OCR
local sobre el VLM de Ollama. `Perception` es inmutable y tiene `merge()`
(combinar resultados de varios motores) y `summary()` (texto plano para agregar).

`EngineError` es la excepción de dominio (conexión, timeout, respuesta
inválida).

### `ENGINE_FACTORY`

`core/engine_manager.py` registra cada motor en un mapa `id -> clase`:

```python
ENGINE_FACTORY = {
    "ollama": OllamaEngine,
    "ollama_vision": OllamaVisionEngine,
    "comfyui": ComfyUIEngine,
    "bfl": BFLApiEngine,
    "wanvideo": WanVideoEngine,
    "piper_tts": PiperTTSEngine,
    "kokoro_tts": KokoroTTSEngine,
    "whisper": WhisperEngine,
    "musicgen": MusicGenEngine,
    "yolo": YoloEngine,
    "clip": ClipEngine,
    "paddle_ocr": PaddleOcrEngine,
}
```

`EngineManager` carga `engines.json`, construye las instancias y expone
`get_engine(id)` (instancia construida y habilitada).

### Motores actuales

| Motor | Responsabilidad | Backend / modelos |
| ------- | ----------------- | ------------------- |
| `ollama` | Chat | Ollama (qwen3.6:35b por defecto, deepseek-r1, …) |
| `ollama_vision` | OCR / resumen / tablas | Ollama (qwen2.5vl:7b) |
| `comfyui` | Imágenes | ComfyUI (Flux Dev / Schnell / Z-Image / Kontext) |
| `bfl` | Imágenes | Black Forest Labs API de pago (FLUX.1 Kontext Pro/Max, FLUX.2 Pro) |
| `wanvideo` | Vídeo T2V/I2V | ComfyUI (Wan 2.2 + Lightx2v) |
| `piper_tts` | TTS rápido | ONNX (6 voces ES/EU) |
| `kokoro_tts` | TTS de calidad | PyTorch CPU (ef_dora, em_alex, em_santa) |
| `whisper` | Transcripción a SRT | faster-whisper (CTranslate2) |
| `musicgen` | Música a partir de texto | audiocraft / Meta MusicGen (small/medium/large; opcional) |
| `yolo` | Detección de objetos | ultralytics (`yolov8*.pt` o un motor `.engine`/`.plan`) |
| `clip` | Etiquetado zero-shot | transformers + torch (vocabulario cerrado) |
| `paddle_ocr` | OCR local | PaddleOCR (DBNet+CRNN), idioma `latin` para es/eu |

## 4. Arquitectura de tareas

### Interfaz `Task`

`tasks/base_task.py`:

```python
class Task(ABC):
    id            -> str
    nombre        -> str
    build_widget(parent) -> QWidget
```

### `TASK_FACTORY`

`core/task_manager.py` registra cada tarea en un mapa `id -> factory`. Cada
factory recibe las dependencias compartidas (`TaskDeps`: `engine_manager` y
`api_manager`) y devuelve la tarea construida:

```python
@dataclass(frozen=True)
class TaskDeps:
    engine_manager: EngineManager
    api_manager: ApiManager

TASK_FACTORY: dict[str, TaskFactory] = {
    "chat": _chat_factory,            # recibe además el orquestador
    "document": _engine_only(DocumentTask),
    "image": _engine_only(ImageTask),
    "video": _engine_only(VideoTask),
    "understanding": _engine_only(UnderstandingTask),
    "audio": _engine_only(AudioTask),
    "assembler": _engine_only(AssemblerTask),
}
```

`_engine_only(cls)` adapta las tareas cuyo constructor solo pide el motor; una
tarea que necesite otra dependencia (p. ej. `ApiManager`) define su propia
factory y la toma de `TaskDeps`, sin romper el bucle uniforme de registro.
`TaskManager(engine_manager, api_manager=None)` crea el `ApiManager` si no se
inyecta y lo expone en `TaskManager.api_manager`.

Cada tarea construye su widget (`ui/<nombre>_widget.py`) y se muestra como
**pestaña** en el `QTabWidget` de `MainWindow`. El orden de pestañas es el
orden de registro en `TASK_FACTORY`.

Cada tarea declara además `Task.engine_id` (motor cuyo modelo se descarga al
abandonar su pestaña); `MainWindow` deriva de ahí su mapa pestaña→motor, de modo
que no hay que mantener una lista paralela al registro de tareas.

## 5. Configuración

`core/config_manager.py` centraliza la configuración en 3 archivos JSON dentro
de `config/`:

- `engines.json` → lista de motores.
- `apis.json` → APIs externas (aún sin usar).
- `settings.json` → `log_level`, `last_models` (persistencia de UI),
  `unload_on_tab_change`, `gpu_scheduler` (ver §9) y `output_limpieza`
  (retención de `output/`: 500 ficheros, 90 días, 8 GB y 24 h de gracia por
  defecto; `MainWindow._cleanup_output` lo aplica al arrancar, en el pool y sólo
  sobre extensiones que escribe la app).

Cada archivo se valida con un esquema propio (`ENGINE_SCHEMA`, `API_SCHEMA`,
`SETTINGS_SCHEMA`). Los archivos reales (`*.json`) están **ignorados por git**;
solo se versionan las plantillas (`*.example.json`).

Los secretos se leen **siempre** de variables de entorno vía `.env`
(`_ensure_dotenv()`): el campo `key` de `apis.json` está **deprecado** y se
ignora (`resolve_api_key` no lo usa), para no guardar claves en claro en disco.
La carga de `settings.json`/`engines.json` es tolerante a fallos: un JSON
corrupto cae a los valores por defecto en lugar de impedir el arranque.

## 6. Flujo de una generación (ejemplo: vídeo I2V)

1. El usuario selecciona una imagen + prompt en `VideoWidget`.
2. `VideoWorker` (`QThread`) llama a `WanVideoEngine.generate_video()`.
3. `WanVideoEngine`:
   - Sube la imagen a ComfyUI (`POST /upload/image`).
   - Carga el workflow (`t2v` o `i2v`) y lo parchea con imagen/prompt.
   - Envía `POST /prompt`.
   - Espera por el **WebSocket `/ws`** (fuente única de verdad):
     `execution_start` → progreso → `execution_success`.
   - Lee la salida una vez tras el éxito (`GET /history/{id}`).
   - Descarga el MP4 (`GET /view`).
4. Al terminar, libera VRAM con `POST /free` (`unload_models`, `free_memory`).
5. El MP4 se guarda en `output/`.

## 7. Convenciones

- Un motor = un archivo en `engines/`.
- Una tarea = un archivo en `tasks/` + un widget en `ui/`.
- La lógica pura (presets, validaciones, preferencias, cálculos) vive en
  `services/`, sin Qt, para poder testearla sin GUI.
- Los widgets de tarea heredan de `ui/task_widget.py` y declaran sus controles
  en `_busy_widgets()`/`_busy_only_widgets()` en vez de reimplementar
  `_set_busy()`; usan `engines.base_engine.CANCEL_MESSAGE` para la cancelación.
  Declaran además su campo de escritura en `_focus_widget()`: la base le
  devuelve el foco al acabar la tarea y al entrar en la pestaña (deshabilitar el
  control lo pierde y Qt no lo recupera solo).
- Un cambio = un commit.
- Verificación automática: `python -m ruff check .` (`ruff.toml`: solo reglas de
  errores reales —sintaxis, nombres/imports sin usar, `except` ciego—, sin reglas
  de estilo) + `python -m compileall -q .` + `python -m unittest discover -s tests`,
  con las versiones de test congeladas en `constraints-test.txt` (CI en
  `.github/workflows/ci.yml`). Además, arranque de la app + prueba manual.
- Las peticiones de red no deben ejecutarse en el hilo de la UI (usar
  `QThread`/`QThreadPool`); el logging se configura desde `settings.json`
  (`log_level` y `rutas.logs`).
- **F5-TTS**: intentado y descartado (watermarking ElevenLabs + sin euskera).
- **Kokoro**: sin euskera (solo cubre 9 idiomas); para voces en euskera se usa
  Piper. Es una limitación **solo de TTS**: los modelos locales de texto y visión
  de Ollama (`qwen3-vl`, familia `qwen3.x`) sí trabajan en euskera, y los prompts
  de Documentos piden conservar el idioma original y responder en él.

## 8. Estado actual

- 7 pestañas: Chat, Documentos, Imagen, Vídeo, Comprensión, Audio, Ensamblador.
- 21 tags (de v0.1-skeleton a v0.14-kokoro).
- Sin refactor de configuración: se mantiene la arquitectura legacy
  (`config/*.json`) y se descartó el intento de config unificada
  (`~/.imperator/config.json`).
- `bfl` incluido en `DEFAULT_ENGINES` (antes solo existía en el factory).
- Suite de tests (stdlib `unittest`, 620 tests) + CI en GitHub Actions (con
  `ruff` y versiones congeladas en `constraints-test.txt`). Incluye
  un smoke test de la `MainWindow` opt-in (`IMPERATOR_GUI_SMOKE=1`), que el CI
  activa. Las suites nuevas (`test_gpu_scheduler`, `test_intent_router`,
  `test_perception_engine`, `test_video_understanding`) usan dobles: no
  necesitan GPU, ni ffmpeg, ni las dependencias pesadas de percepción.
- El CI instala `requirements-test.txt` (PyQt6, requests, python-dotenv, numpy),
  no todo `requirements.txt`: los imports de los motores pesados están
  protegidos con `try/except ImportError`.
- `CANCEL_MESSAGE` es única (`engines/base_engine.py`) para motores y widgets,
  con un test-guarda que impide volver a duplicar el literal.
- Las llamadas a `ffmpeg` (`core/video_assembler.py`) llevan `timeout` y kill:
  un ffmpeg atascado ya no puede colgar el hilo del worker.
- Motores divididos por familias (`Chat`/`Vision`/`Image`/`Video`/`Audio`/
  `Music`/`Transcription`) en `engines/base_engine.py`.
- El chat renderiza markdown (`ui/markdown.py`): negritas, tablas y listas como
  texto enriquecido, con el código en monoespaciado y con los colores del tema.
- Documentos puede resumir el documento **completo** (una sola llamada al
  modelo, en vez de página a página) con la casilla "Resumen global". Los
  resúmenes (por página y globales) comparten reglas en
  `document_params.SUMMARY_STYLE`: un bloque por tema con **título tomado de los
  epígrafes del documento** (en texto plano: el área de resultado no pinta
  markdown), frases completas de longitud parecida, sin repetir datos y
  **fidelidad al texto** (sin interpretar, inferir ni inventar). La pestaña tiene
  dos casillas para el resumen, ambas persistidas y activas solo en ese modo:
  **alcance global** (una sola llamada con el documento entero) y **frase de
  cierre por apartado** (por defecto apagada; encendida, `CLOSING_STYLE` la pide
  extractiva —solo repite lo que ya dice el bloque— porque pedirla a secas hacía
  que el modelo inventara datos para "concluir").
- El modelo por defecto del chat es `qwen3.6:35b`, definido en un solo sitio
  (`_OLLAMA_DEFAULT_MODEL`) y replicado en la configuración y la UI, con un
  test-guarda (`test_ollama_defaults.py`) que impide que se desincronicen.
- Los motores de Ollama piden ventana de contexto explícita y esta se puede fijar
  por motor con `num_ctx` en `engines.json`: 8192 en el chat
  (`ollama_api.DEFAULT_NUM_CTX`) y 16384 en la visión
  (`DEFAULT_VISION_NUM_CTX`, porque ahí el `thinking` consume ventana antes de
  escribir la respuesta). Sin pedirla, Ollama usaba 4096 y el análisis de un
  documento fallaba con HTTP 400 ("exceeds the available context size"); ese 400
  ahora se traduce a un mensaje que dice qué hacer (`ollama_api.http_error_message`).
- El análisis de documentos tiene timeout propio (600 s por defecto,
  `parametros.timeout`) porque los modelos de visión con razonamiento gastan ~90%
  de los tokens en `thinking` (medido con `qwen3-vl:8b`: 4143 tokens generados
  para 390 caracteres de respuesta), y ese razonamiento **llena la ventana que se
  le dé** (2000 tokens de `thinking` con 4096 de contexto, 10 300 con 16384). Por
  eso la visión pide 16384: una página de ~9800 caracteres (4303 tokens de prompt)
  se resume en ~113 s, mientras que a 4096 se quedaba sin respuesta.
  `think: false` no lo desactiva en qwen3-vl y `num_predict` los deja sin
  respuesta, así que la respuesta cortada por contexto y la respuesta vacía por
  razonamiento se avisan explícitamente.
- El procesado de documentos va por lotes (`ui.document_widget.MAX_PARALLEL_PAGES
  = 2`): solapar páginas solo compensa si Ollama trae varios slots
  (`OLLAMA_NUM_PARALLEL=2` y reiniciar el servidor), porque con uno las
  peticiones se encolan y el paralelismo solo añade sobrecarga (medido **0,84x**).
  Con dos slots: **1,68x** en peticiones crudas y **1,32x** en el camino real con
  3 páginas. Como Ollama **reparte el contexto entre slots**, cada motor pide el
  doble: visión 32768 (16384 por página) y chat 16384 (los 8192 por petición de
  antes).
- Documentos sigue el **pipeline por fases** de `services/document_pipeline.py`
  (FASE 0 detección → FASE 1 OCR/tablas → FASE 2 normalización y resumen →
  FASE 4 validación ligera → FASE 5 entrega): el módulo puro decide el plan
  (`PipelinePlan`, `route`, `phases`, `tables_extra`, y los prompts) y
  `ui.document_widget.DocumentWorker` lo ejecuta con un método por fase, de modo
  que el código se lee igual que el diagrama. Dos casillas nuevas, ambas
  persistidas: **«Forzar OCR»** (transcribir con el modelo aunque el PDF ya traiga
  capa de texto) y **«Extraer tablas además»** (bloque `--- Tablas (Página N) ---`
  junto al resultado principal —transcripción o resumen—, reutilizando el texto ya
  extraído en vez de volver a rasterizar). La FASE 4 no bloquea nada: solo añade
  avisos (documento recortado, entrega vacía) y la FASE 5 es el único sitio que
  monta el texto final. El alcance del resumen es un selector
  (**Automático / Global / Por página**): el automático resume el documento
  completo en una sola llamada a partir de
  `document_params.AUTO_GLOBAL_MIN_PAGES` (6) páginas, porque medido con
  `qwen3-vl:8b` una llamada por página cuesta ~50 s (19 páginas ≈ 16 min) y el
  documento entero ~30-48 s; el motivo del coste por página es el razonamiento
  del modelo (~5 900 tokens generados para ~650 caracteres de respuesta).
- El resumen global **no recorta** el documento: si no cabe en una consulta se
  resume por bloques (`document_pipeline.summary_blocks`, que corta por párrafos
  y por frases completas) y después se resume el conjunto, así que ninguna
  página se queda fuera. Recortar por el final era peor que perder texto:
  medido, el modelo **extrapolaba** lo que no veía (deducía la serie aritmética
  del documento y se inventaba los valores de las páginas ausentes). Por eso la
  FASE 4 comprueba las cifras del resultado contra el documento
  (`document_pipeline.check_fidelity`) y avisa de las **inventadas** (un error
  real); de las omitidas no avisa, porque un resumen es selectivo por
  definición. El **idioma** no se le deja al modelo:
  `document_params.detect_language` lo detecta en el documento (palabras
  funcionales comparadas por prefijo, útil para el euskera) y el worker lo fija
  en el plan con `PipelinePlan.with_language`, de modo que el prompt lo pide por
  su nombre; con «responde en el mismo idioma» el modelo traducía (documento en
  euskera → resumen en castellano con los títulos en euskera). Un idioma
  elegido a mano nunca se pisa.
- `musicgen` es el motor de música: `audiocraft` es opcional, así que sin él la
  app arranca igual y el motor se marca como no disponible.
- `ApiManager` (`core/api_manager.py`) da un consumidor real a `apis.json`.
- `core/gpu_scheduler.py` decide **si** un trabajo pesado puede empezar, con
  tres prioridades (`perception` > `interactive` > `batch`) y VRAM medida de
  verdad con `nvidia-smi`. Sustituye el «a ojo» por datos, es thread-safe y
  **falla abierto**: sin `nvidia-smi` nunca bloquea (un planificador que rompiera
  la app por no poder medir sería peor que no tenerlo). Se configura en
  `settings.json` → `gpu_scheduler`. La ventana lo consulta en
  *Motores → Estado de la GPU…* (el sondeo va al pool, no al hilo de la UI).
- **Quién lo usa: las cinco tareas pesadas.** Chat, Documentos, Imagen y Vídeo
  reservan GPU con `ui.workers.gpu_slot` (y Comprensión, que además reserva por
  cada fotograma de percepción con `scheduler.job(...)`). El contexto pide permiso
  al planificador y, si no hay hueco, **pide a los demás motores que suelten su
  VRAM y reintenta una vez** (el mismo remedio que *Liberar VRAM ahora*, pero
  automático); si aun así no cabe, la tarea falla con el motivo en pantalla en
  vez de morir por OOM a mitad de trabajo.
  La reserva **no declara el consumo de cada modelo** (`vram_mb=0`) a propósito:
  Ollama y ComfyUI reparten capas entre CPU y GPU, así que una estimación alta
  bloquearía trabajos que hoy funcionan; lo que se evita es *empezar* con la GPU
  casi llena o con otro lote en marcha, y para eso basta el margen
  (`vram_headroom_mb`) y la regla de lote ocupado. Medido con la GPU real
  (RTX 5080, 16 303 MB): Flux Dev 1024×1024/20 pasos pica en **15 438 MB**, el
  chat `qwen3.6:35b` ocupa **~13 GB** (13 116 MB medidos, con `num_ctx` 16 384) y
  la visión `qwen3-vl:8b` a 16 384 **10 210 MB**. Con el chat cargado la GPU se
  queda con **619 MB libres** y el planificador ya rechaza cualquier trabajo
  («quedan 619 MB y se necesitan 1 024 MB»): es exactamente el caso que antes
  acababa en OOM y que ahora se resuelve pidiendo al otro motor que suelte su
  VRAM (comprobado: tras liberarla, hay 14 763 MB y el trabajo arranca).
  `max_concurrent` sigue siendo configurable: con `1` no se solapan dos lotes
  (lo recomendable en esta GPU, donde dos modelos grandes no caben a la vez).
- Correcciones de la auditoría del 30/09/2026 (todas con medición detrás):
  las URL de los motores locales usan `127.0.0.1` y no `localhost` (`localhost`
  añadía ~2 s por petición por el fallback de IPv6 de Windows: 2 040 ms frente a
  15 ms); `MAX_PARALLEL_PAGES` vuelve a `1` y la visión a 16 384 tokens (medido
  con el worker real: 1,02x y 0,62x, el paralelo no compensaba); el progreso de
  Documentos se emite **por página** (con lotes sólo llegaba el aviso del final
  del lote); el modelo por defecto de `OllamaEngine` dejó de estar desincronizado
  (`qwen3.5:9b` → `qwen3.6:35b`, con test-guarda); los `raise_for_status()`
  sueltos de ComfyUI/BFL se traducen a `EngineError`; las descargas de resultados
  tienen tope (`base_engine.MAX_DOWNLOAD_BYTES`, 512 MiB) y `ConfigManager`
  serializa las escrituras (lock + temporal único + `update_settings`).
- **El arranque ya no paga los motores que no se usan.** `kokoro` (9,53 s: arrastra
  `transformers` + `torch`) y `perception` (8,72 s: `torch` + `transformers`) se
  importan **al usar el motor**, y su `is_available()` se resuelve con
  `importlib.util.find_spec` (~0 ms). Medido: `import ui.main_window` pasó de
  **9,85 s a 0,46 s**, el total hasta tener la ventana de **~11,9 s a ~0,54 s** y
  la RAM en reposo de 618 MB a 193 MB (torch/transformers ya no se cargan).
  El precio es explícito: la **primera** síntesis con Kokoro de cada sesión paga
  su import (~10 s), que antes pagaba todo el mundo al arrancar.
- **Retención de salidas.** `services/output_cleanup.py` limita `output/` (500
  ficheros, 90 días, 8 GB y 24 h de gracia; configurable en `settings.json` →
  `output_limpieza`) y se aplica al arrancar en el pool. Sólo mira extensiones que
  escribe la app, así que lo que no sea un producto de Imperator se respeta.
- `services/intent_router.py` es el router de intención del chat: reglas léxicas
  deterministas (es/eu/en, sin LLM) que deciden a qué tarea apunta un mensaje.
  Exige **señal compuesta** (umbral 2.5): «pdf» a secas no enruta, «resume este
  pdf» sí, así no secuestra conversaciones normales.
  `services/task_orchestrator.py` resuelve esa intención contra las tareas
  registradas; `TaskManager` lo inyecta en el chat por `TaskDeps`, y el chat se
  limita a **emitir** (`routed`) mientras `MainWindow` ofrece abrir la pestaña.
- `engines/perception.py` añade percepción estructurada: `yolo` (detección),
  `clip` (etiquetado zero-shot) y `paddle_ocr` (OCR local). Los tres devuelven
  `Perception` (objetos/etiquetas/texto), no lenguaje natural, y sus dependencias
  pesadas van con import protegido: sin ellas la app arranca igual y el motor
  aparece como no disponible (mismo criterio que `musicgen`). Vienen
  **deshabilitados** por defecto en `engines.json`.
  Aviso de idioma: no hay modelo de euskera propio en PaddleOCR (se usa `latin`)
  ni CLIP describe en abierto (puntúa un vocabulario cerrado).
- `tasks/understanding.py` + `ui/understanding_widget.py` añaden la pestaña
  **Comprensión**: muestrea fotogramas del vídeo (uno cada N segundos **en el
  centro** de cada intervalo, con tope de fotogramas — «cada N ms» es inviable),
  los pasa por los motores de percepción, extrae el audio y lo transcribe con
  Whisper, resume el conjunto con Ollama y monta el informe. El plan y el
  formato viven en `services/video_understanding.py` (puro) y el ejecutor es
  `UnderstandingWorker`, igual que Documentos. Si falta un motor, el informe lo
  **avisa** en vez de rellenar el hueco inventando.

## 9. Coordinación de GPU y orquestación

Dos piezas resuelven, respectivamente, «quién usa la GPU» y «qué tarea responde
a este mensaje»:

- **Prioridades de GPU.** `perception` (OCR de una página, un fotograma),
  `interactive` (lo que el usuario acaba de pedir: chat, resumen) y `batch`
  (imagen, vídeo, comprensión completa). Un trabajo de lote **cede** si la GPU
  está ocupada (`defer_batch_above_util`, `defer_batch_below_free_mb`) o si no
  queda VRAM para su estimación más una reserva (`vram_headroom_mb`).
- **Motores `.plan` (TensorRT): posible, pero solo en los motores nuevos.** Los
  `*.plan`/`.engine` no gobiernan Ollama ni ComfyUI: son runtimes externos
  (llama.cpp/CUDA y PyTorch/CUDA). Un motor compilado solo se puede usar donde
  hay una inferencia propia que lo cargue; en esta arquitectura eso es `yolo`
  (Ultralytics acepta la ruta del motor) y, opcionalmente, `clip`/`paddle_ocr`.
  Ollama y ComfyUI siguen por HTTP, y el planificador se limita a decidir *si*
  se les lanza trabajo.
- **Orquestación del chat.** El chat no abre pestañas ni conoce widgets: emite
  una intención (`routed`) y `MainWindow` decide (hoy pregunta al usuario). El
  punto de extensión para una ejecución automática futura es
  `services/task_orchestrator.py`, no el widget de chat.
