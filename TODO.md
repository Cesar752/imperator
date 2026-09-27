# TODO — PlataformaIA

Notas pendientes (revisar al retomar el proyecto).

## 1. Verificar ducking con MP3 reales (≈5 min)

**Estado:** pendiente — falta material de prueba.

- El ducking ya está implementado en `core/video_assembler.py` (`_audio_filter_complex`),
  usando `sidechaincompress=threshold=0.05:ratio=8:attack=10:release=250`:
  la voz (`[vsc]`) comprime la música para bajarla mientras hay locución.
- La música de fondo se elige en `ui/assembler_widget.py` (carpeta `music/`, `MUSIC_DIR`).
- **Bloqueo actual:** `music/` está vacío (0 archivos). Falta meter MP3 reales.

**Al tener música en `music/`:**

1. Añadir uno o varios MP3 reales a `C:\Imperator\music\`.
2. En la pestaña Ensamblador: cargar un vídeo + una voz + música de fondo,
   activar "Ducking", ensamblar.
3. Comprobar que:
   - La música suena de fondo y baja de volumen durante la voz.
   - La música no se corta al final (usa `-stream_loop -1` + `-shortest`).
   - Sin ducking, la música se mantiene a volumen constante (`music_volume`).
4. Probar también el caso "solo música" (sin voz) y el caso con subtítulos
   (el `subtitles` vive dentro del mismo `-filter_complex`).

## 2. Fase 13e (opcional): MusicGen local

**Estado:** hecha — ver §5.3.

- `engines/musicgen.py` genera música con
  [Meta MusicGen](https://github.com/facebookresearch/audiocraft).
- Sigue el patrón de `engines/base_engine.py` (`id`, `nombre`, `is_available()`,
  `list_models()`) y añade `generate_music(prompt, model, duration,
  output_path)`.
- **Nota:** el aviso de que `Engine` obligaba a `chat()` y había un
  `TODO(fase N)` estaba **desfasado**: el ABC ya estaba dividido por familias.
  Lo que faltaba era la familia `MusicEngine` (añadida en §5.3).
- Registrado en `core/engine_manager.py` (+ `DEFAULT_ENGINES` y plantillas de la
  UI de motores).

## 3. Punto de retorno de la sesión (2026-09-27)

**Para retomar:** proyecto Imperator (`C:\Imperator`, PyQt6 + Ollama + ComfyUI).
Sesión del 2026-09-27: refactor a `services/`, foco y formato del chat, idioma de
salida en Documentos, límites de Ollama y descarga de VRAM. 245 tests en verde.
Siguiente: markdown en el chat y resumen global de documentos.

**Qué se hizo** (commits `c8748b8` … `201b99a`):

- `services/`: nuevos `audio_params.py`, `document_params.py`, `chat_params.py` y
  `system_stats.py`; `preferences.py` ampliado. Todo con tests.
- `ui/task_widget.py`: base común de los 6 widgets (foco, estado y ciclo del
  worker); `_focus_widget()` devuelve el cursor al campo de escritura.
- Chat: turnos con autor resaltado (el **modelo**, no "Ollama"), scroll
  automático, sin turnos vacíos y con aviso si el motor no responde.
- Documentos: el prompt del modo se aplica también al texto nativo de un PDF;
  aviso si el modelo elegido no cubre el euskera; **selector de idioma de salida**
  (`auto`/`es`/`eu`/`en`, solo para resumen y tablas; el OCR nunca traduce).
- Ollama: `num_ctx=8192`, sin tope de `num_predict`, aviso cuando la respuesta se
  trunca, y descarga de VRAM **solo** de lo que está cargado (`/api/ps`).
- `ffmpeg` con `timeout` y kill; `pyrightconfig.json`; CI con
  `requirements-test.txt`.

**Verificación rápida** (245 tests, 1 skip):

```
venv\Scripts\python.exe -m unittest discover -s tests -t .
```

**Candidatos para la próxima sesión:**

1. Renderizar markdown en las respuestas del chat (hoy se ven `**` y las tablas
   en crudo); requiere `QTextEdit` + `QTextDocument.setMarkdown()` y traducir los
   estilos al tema oscuro.
2. Resumen de documento completo en vez de página a página (opción en la UI).
3. ~~Renderizar markdown en las respuestas del chat.~~ **Hecho** — ver §5.1.
4. ~~Resumen de documento completo en vez de página a página.~~ **Hecho** — ver §5.2.
5. ~~`TODO(fase 6)` de `core/task_manager.py`: dependencias de tareas por `kwargs`.~~
   **Hecho** — ver §4. Ahora son factories con `TaskDeps`, no `kwargs`.
6. Los MP3 en `music/` para validar el ducking (ya hay 3 MP3 en `music/`; ver §1).
7. ~~Fase opcional: motor local MusicGen.~~ **Hecho** — ver §5.3.

## 5. Sesión de continuación (2026-09-27): chat, documentos y música

Se abordaron los tres candidatos que quedaban del §3 (§1 sigue pendiente: probar
el ducking con los MP3 reales). **297 tests en verde** (1 skip).

### 5.1 Markdown en el chat

- `ui/markdown.py` (nuevo): `build_html(markdown, dark)` importa el markdown con
  `QTextDocument.setMarkdown()` y traduce al tema lo que Qt deja sin tratar: los
  bloques de código (los marca con `nonBreakableLines`) y el código en línea
  (`fontFixedPitch`) reciben monoespaciado, fondo y color según claro/oscuro. El
  texto sin marcadores se inserta tal cual, sin pasar por HTML.
- `ui/chat_widget.py`: el área de conversación pasa de `QPlainTextEdit` a
  `QTextEdit`; el cuerpo de cada turno lo pinta `md.insert_markdown()` (Qt fusiona
  el primer bloque con el párrafo en curso, así que la etiqueta del autor sigue
  en su línea).
- **Ojo con el scroll:** `ensureCursorVisible()` en `QTextEdit` deja el final a un
  margen de documento (4 px) de distancia; `_scroll_to_bottom()` fuerza además la
  barra a su máximo (los tests de scroll de `test_chat_widget` lo cubren).
- `tests/test_markdown.py`: 10 tests.

### 5.2 Resumen global de documento

- `services/document_params.py`: `GLOBAL_MODES`, `supports_global`,
  `wants_global`, `GLOBAL_PROMPT` + `global_prompt_for` (respeta el idioma de
  salida), `merge_pages` (une las páginas y dice si hubo que recortar),
  `MAX_GLOBAL_CHARS` y `TRUNCATED_NOTICE`.
- `ui/document_widget.py`: `DocumentWorker(..., global_summary=)` hace dos
  pasadas —transcribe cada página (capa nativa u OCR, con el prompt de OCR) y
  resume el conjunto en una única llamada— y la pestaña gana la casilla
  "Resumen global (documento completo)", activa solo en modo Resumen y
  persistida en `prefs.DOCUMENT_GLOBAL`.
- Reglas del resumen (`SUMMARY_STYLE`), en tres rondas de revisión con
  documentos reales. Valen para el resumen por página y para el global; el OCR y
  las tablas no las llevan.
  - Ronda 1: un párrafo corto por apartado (sin mezclar temas), frases de
    longitud parecida y sin repetir datos ya dados en otro apartado.
  - Ronda 2: **se quitó el cierre por apartado con "→"** que se había pedido en
    la ronda 1. Invitaba a *interpretar* y el modelo llegó a inventar datos
    (hablaba de "equipamiento" donde el documento decía acumular la hora de
    lactancia). Ahora lleva reglas de **fidelidad**: resumir solo lo que dice el
    documento, sin interpretar, inferir, opinar ni inventar términos o cifras, y
    sin líneas de cierre ni flechas.
  - Ronda 3: **títulos por bloque** (sin ellos el resumen se lee como un texto
    corrido y cuesta localizar cada apartado). Se piden en texto plano y tomados
    de los epígrafes del propio documento, sin inventarlos, porque el área de
    resultado de Documentos no pinta markdown (mismo motivo por el que el chat,
    que sí lo pinta, tiene su propio módulo `ui/markdown.py`).
- `tests/test_document_global.py`: 14 tests; las reglas del resumen se cubren con
  6 tests en `tests/test_services.py`.

### 5.4 Modelo por defecto del chat: qwen3.6:35b

El default era `qwen3.5:9b` y la preferencia guardada apuntaba a `qwen3.8:27b`
(que ya no está en Ollama): de ahí el aviso "Modelo guardado 'qwen3.8:27b' ya no
existe en Ollama; usando el primero del combo". El aviso en sí es correcto como
red de seguridad, así que se mantiene, pero ahora no salta.

- El valor vive en cuatro copias y **las cuatro** pasan a `qwen3.6:35b`:
  `_OLLAMA_DEFAULT_MODEL` (`core/engine_manager.py`), `DEFAULT_ENGINES`
  (`core/config_manager.py`), la plantilla de `ui/engines_dialog.py` y
  `config/engines.example.json`.
- `tests/test_ollama_defaults.py` (2 tests) evita que vuelvan a desincronizarse.
- Actualizados también los ficheros **locales** (no versionados), para que el
  cambio surta efecto en el próximo arranque: `config/engines.json`
  (`ollama.parametros.model`) y `config/settings.json` (`last_models.chat`, que
  era `qwen3.8:27b`).
- `qwen3.6:35b` está entre los modelos instalados en Ollama (comprobado con
  `/api/tags`), así que el combo lo selecciona sin avisos.

### 5.3 Motor local MusicGen

- `engines/base_engine.py`: nueva familia `MusicEngine` con
  `generate_music(prompt, **kwargs) -> Path`.
- `engines/musicgen.py`: motor opcional (import de `audiocraft` protegido),
  tamaños `small`/`medium`/`large`, duración acotada a 1-30 s, modelo cacheado y
  escritura del WAV con `audio_write`.
- Registro: `ENGINE_FACTORY` + rama en `EngineManager._build`, entrada en
  `DEFAULT_ENGINES` y en `config/engines.example.json`, y plantillas de
  `ui/engines_dialog.py` (un test-guarda comprueba que la UI cubre todos los
  tipos del factory).
- `requirements.txt`: `audiocraft` queda comentada como dependencia opcional.
- `tests/test_musicgen_engine.py`: 15 tests con dobles del modelo.

## 4. TODO(fase 6) resuelto: factories con dependencias (2026-09-27)

`core/task_manager.py` asumía `cls(engine_manager)` para todas las tareas: al
añadir una que necesitara otra dependencia (p. ej. `ApiManager`) había que romper
la construcción uniforme. Migrado a factories que reciben un contenedor:

- `core/task_manager.py`: nuevo `TaskDeps` (frozen) con `engine_manager` y
  `api_manager`; `TASK_FACTORY` pasa de `dict[str, type[Task]]` a
  `dict[str, TaskFactory]`. `_engine_only(cls)` adapta las tareas que solo piden
  el motor; una tarea con más dependencias usa su propia factory.
- `TaskManager(engine_manager, api_manager=None)`: si no se inyecta, crea el
  `ApiManager` con el `ConfigManager` del `EngineManager`; expone
  `TaskManager.api_manager`.
- `ui/main_window.py`: comparte su `ApiManager` con el `TaskManager` (una sola
  lectura de `config/apis.json`).
- `tests/test_task_manager.py` (6 tests): orden/ids de las tareas, contrato
  `Task`, `get_task`, `ApiManager` inyectado vs. creado, factory con
  dependencias extra y `TaskDeps` inmutable.
