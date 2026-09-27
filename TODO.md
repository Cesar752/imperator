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
  - Ronda 4: se **refuerzan** A y B ("usa *todos* los epígrafes del documento,
    uno por bloque" y "frases completas de longitud parecida, no fragmentos
    telegráficos") y el cierre vuelve como **opción** (abajo).
- Las reglas y las dos casillas del resumen se cubren en
  `tests/test_services.py` (prompts) y `tests/test_document_global.py` (worker y
  widget).

#### Frase de cierre por apartado: opcional y extractiva

La ronda 2 quitó el cierre con "→" porque el modelo lo usaba para interpretar
(inventó datos). La ronda 4 lo devuelve como **casilla** —"Frase de cierre por
apartado (→)", apagada por defecto— para no perder ninguna de las dos cosas:

- **Apagada** (por defecto): `NO_CLOSING_STYLE` prohíbe cierres y flechas. Es el
  modo "de examen": todo lo que sale está en el documento.
- **Encendida**: `CLOSING_STYLE` pide una línea de cierre que recoja lo que **ya
  dice ese bloque** ("quote or rephrase its own words: never add claims, opinions,
  advice or data that the block does not state").

Medido con la misma página y `qwen3-vl:8b` (57 s con la casilla encendida), el
cierre sale así:

```
Vacaciones
Hilabetean 2,5 egun metatzen dira, gehienez 30 egun urtean.
→ Hilabetean 2,5 egun metatzen dira, urtean gehienez 30 egun.
```

Con la restricción de fidelidad el cierre **no inventa** (objetivo cumplido),
pero queda casi como una repetición del propio apartado. Es el compromiso real:
un cierre que *aporte* algo nuevo es, por definición, una interpretación (lo que
la ronda 2 descartó). Las dos casillas del resumen (alcance global y cierre) solo
se activan en el modo Resumen y se persisten en `settings.json`.

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

### 5.5 El 400 de Ollama en Documentos: el motor de visión no pedía contexto

**Síntoma:** al procesar un documento, la tarea moría con "Error al hablar con
Ollama: 400 Client Error: Bad Request for url: <http://localhost:11434/api/chat>".
El motivo real solo se veía en el log de Ollama:

```
srv send_error: task id = 0, error: request (4135 tokens) exceeds the available
context size (4096 tokens), try increasing it
```

**Causa:** `OllamaVisionEngine._chat()` montaba el payload **sin `options`**, así
que Ollama usaba su contexto por defecto (4096). Una página con capa de texto —y
sobre todo el resumen global, que manda hasta `MAX_GLOBAL_CHARS` (12000
caracteres, y en euskera se tokeniza peor)— no cabía. El motor de chat sí pedía
`num_ctx`; por eso solo fallaba Documentos.

- `engines/ollama_vision.py`: añade `options.num_ctx` al payload y el contexto es
  configurable (`_NUM_CTX`, por defecto el compartido).
- `engines/ollama_api.py`: `DEFAULT_NUM_CTX = 8192` como única fuente para chat y
  visión, y `http_error_message()`, que lee el motivo del cuerpo de la respuesta
  y, cuando es el contexto, lo explica y dice qué tocar (`num_ctx` en
  `engines.json` y/o `OLLAMA_CONTEXT_LENGTH` en el servidor). Antes se mostraba
  el texto genérico de `raise_for_status`, que no decía nada.
- `core/engine_manager.py`: `num_ctx` (y `num_predict` en el chat) se pueden fijar
  por motor en `engines.json`; si no están, mandan los valores del motor.
- **Verificado contra Ollama** (no solo en teoría): `/api/ps` publica
  `context_length`, y con `num_ctx=8192` en la petición devuelve **8192** tanto
  en `qwen3.5:9b` como en `qwen3.6:35b` (13,3 GB de VRAM), es decir, el servidor
  respeta lo que se le pide. Por defecto servía 4096.
- 9 tests nuevos: payload de visión con contexto, mensaje del 400 (con detalle,
  sin detalle, sin JSON y sin respuesta) y coherencia del valor por defecto entre
  chat y visión.

### 5.6 Timeout en Documentos: el modelo de visión razona antes de responder

Con el 400 resuelto, el siguiente fallo fue "Ollama Vision: timeout al procesar el
documento". El log de Ollama mostraba la petición viva y cortada por nosotros:

```
slot print_timing: id 0 | task 18900 | n_gen = 5173, tg = 29.04 t/s
[GIN] 2026/09/27 - 21:06:30 | 500 |  3m0s | POST "/api/chat"   <- timeout de 180 s
```

**Medido con `qwen3-vl:8b`** (página de 3000 caracteres, prompt de 4026):

| caso | tiempo | tokens generados | «thinking» | respuesta |
| --- | --- | --- | --- | --- |
| normal | 41,5 s | 4143 | 13.859 chars | 390 chars (`stop`) |
| `think: false` | 54,6 s | 6728 | 22.644 chars | 0 (`length`) |
| `num_predict: 700` | 7,3 s | 700 | 2755 chars | 0 (`length`) |

Conclusiones de las medidas:

1. El modelo **razona**: el ~90% de los tokens se van en `thinking`, y por eso una
   página tarda 40 s (o minutos, si el modelo es un 30b) aunque la respuesta final
   sea de 390 caracteres.
2. **`think: false` no lo desactiva** en `qwen3-vl` (empeora: 22.644 caracteres
   pensados y cero respuesta), y un tope de `num_predict` deja la respuesta
   **vacía** porque el razonamiento se come el presupuesto. Ambas cosas quedan
   descartadas con datos, no por intuición.
3. Lo que fallaba de verdad era **nuestro** timeout de 180 s, que cortaba
   peticiones que iban a terminar bien.

Cambios:

- `engines/ollama_vision.py`: timeout de análisis de **600 s** por defecto y
  configurable (`parametros.timeout` en engines.json); aviso en la propia
  respuesta cuando Ollama la corta por contexto; y error explicativo cuando no
  hay respuesta por razonamiento o por contexto lleno (antes se devolvía una
  página en blanco en silencio). Una página realmente sin texto sigue devolviendo
  vacío, que eso no es un error.
- 5 tests nuevos (timeout por defecto y configurado, aviso de respuesta cortada,
  los dos casos de respuesta vacía y la página en blanco legítima).
- **Recomendación** (no medida: ese modelo no está instalado aquí): en Documentos,
  un modelo de visión **sin razonamiento** —p. ej. `qwen2.5vl:7b`, que el propio
  `document_params` reconoce como multilingüe— debería procesar cada página en
  segundos en vez de minutos. `qwen3-vl` entiende más, pero piensa de más.

**Ajuste final, tras seguir fallando con `num_ctx=8192`:** el prompt + el
razonamiento llenaban la ventana y la respuesta llegaba vacía, así que la visión
pasa a pedir **16384** (`ollama_api.DEFAULT_VISION_NUM_CTX`, distinto del chat
porque el motivo no es el prompt sino el `thinking`). Medido con `/api/ps` a
16384:

| modelo | peso | en VRAM | contexto servido |
| --- | --- | --- | --- |
| `qwen3-vl:8b` (por defecto) | 7,2 GB | 7,2 GB (todo) | 16384 |
| `qwen3-vl:30b-a3b` | 19,4 GB | 12,9 GB (resto en RAM) | 16384 |

Conclusiones:

- Con el modelo por defecto (8b), subir la ventana **no cuesta rendimiento**: cabe
  entero en VRAM incluso a 16384.
- El `qwen3-vl:30b-a3b` que había guardado en Documentos **no cabe en 16 GB ni con
  el contexto anterior** (19,4 GB de peso): conviene cambiarlo a `qwen3-vl:8b` en
  el combo de la pestaña.
- En la config local (`config/engines.json`, no versionado) queda explícito
  `ollama_vision.parametros.num_ctx: 16384`.

**¿Y esto lo rompió la ampliación del prompt del resumen?** No. Medido con una
página larga (4460 caracteres), mismo modelo:

| prompt | ctx | tiempo | prompt tokens | generados | «thinking» | respuesta |
| --- | --- | --- | --- | --- | --- | --- |
| viejo (sin reglas) | 4096 | 36,5 s | 1923 | 2173 | 7199 | **0** (`length`) |
| nuevo | 4096 | 57,7 s | 2083 | 2013 | 8077 | **0** (`length`) |
| nuevo | 16384 | 179 s | 2083 | 10300 | 37.337 | 475 (`stop`) |

El prompt viejo **fallaba igual** en esa página: lo que decide si funciona es el
tamaño de la página frente a la ventana, no las reglas de estilo (que añaden
~60 tokens). Y un dato que explica el comportamiento: **el razonamiento llena la
ventana que le des** — 2000 tokens de `thinking` con 4096 y 10 300 con 16384.

Caso idéntico al del usuario (página de ~9800 caracteres → **4303 tokens** de
prompt, frente a los 4135 del log real) con 16384: **113 s, status 200, respuesta
completa (473 caracteres, con títulos y `done=stop`)**, 11 899 de 16 384 tokens
usados. Queda margen para páginas algo mayores.

### 5.7 Motor local MusicGen

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

### 5.8 Procesado por lotes (2 páginas a la vez): medido, apagado por defecto

Idea: solapar páginas aprovechando que Ollama atiende varias peticiones a la vez.
Se implementó en `DocumentWorker` (`parallel=`, lotes con `_chunks`), dejando la
**lectura** de cada página y el montaje del resultado en el hilo del worker
(PyMuPDF no es thread-safe) y conservando orden, progreso y cancelación. Medido
con `qwen3-vl:8b`:

| caso | tiempo | ganancia |
| --- | --- | --- |
| 2 peticiones crudas por HTTP, con dos slots activos | 140 s serie / 83 s paralelo | **1,68x** |
| `DocumentWorker` completo, 3 páginas, carga con **un** slot | 106 s serie / 126 s paralelo | **0,84x** |

La diferencia enseña lo importante: **Ollama decide cuántos slots abre al cargar
el modelo** y el cliente no lo controla. Con un solo slot las peticiones se
encolan y el paralelismo solo añade sobrecarga y contención de caché; con dos, se
solapan de verdad (1,68x).

Decisión: `MAX_PARALLEL_PAGES = 1`, es decir **el mismo comportamiento que antes
de esta sesión** (así nada puede ir más lento), con la función ya escrita y
probada para activarla cuando interese: forzar dos slots en el servidor
(`OLLAMA_NUM_PARALLEL=2` y reiniciar Ollama), subir `num_ctx` de la visión a
**32768** (Ollama reparte el contexto entre slots: 16384 por página) y poner
`MAX_PARALLEL_PAGES = 2`. 6 tests cubren lotes de dos en vuelo, lotes de uno,
orden de las páginas, resultado correcto por página y cancelación entre lotes.

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
