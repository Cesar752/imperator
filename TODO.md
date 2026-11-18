# TODO — PlataformaIA

Notas pendientes (revisar al retomar el proyecto).

## 0. RESUELTO (02/10/2026): error `image_paths`, chat bloqueado y «Fotos primero»

**1. El «error inesperado: ... unexpected keyword argument 'image_paths'»** tenía
una raíz que no era el motor: `video_params.build_generate_params()` metía
`image_paths`/`prompts` **vacíos** cuando se pedía `i2v` sin fotos, que es justo
lo que hace el **encadenado entre escenas** (el `image_path` se pone después, con
el último fotograma). El motor no entiende esas claves y moría en la escena 2.
Corregido en los tres bordes:

- `build_generate_params`: sin fotos **no** se añaden esas claves (test-guarda en
  `tests/test_services.py`).
- `VideoWorker` (pestaña Vídeo): traduce `image_paths`/`prompts` (varias fotos) a
  un `image_path` por llamada; una sola foto también (antes solo se manejaba con
  más de una).
- `WanVideoEngine.generate_video`: acepta `**kwargs` (como manda la interfaz
  `VideoEngine`) y, si llega una clave que no soporta, lanza **error de motor**
  con el nombre del parámetro en vez de un `TypeError` crudo.

**2. El chat se bloqueaba a sí mismo** («quedan 557 MB libres y se necesitan
1024 MB»): el planificador de GPU exigía hueco de VRAM a una tarea cuyo **propio
modelo ya estaba cargado** (`qwen3.6:35b` residente ⇒ 557 MB libres). Arreglado
con `already_loaded` en `GpuScheduler.can_start/acquire` y `gpu_slot`, que lo
resuelve con `engines.ollama.is_loaded()` (vía `/api/ps`) en Chat, Documentos y
el guionista. Además, `gpu_slot` ya **no** pide a los demás motores que suelten
su VRAM si hay un trabajo en marcha (`GpuScheduler.busy`): le rompería el modelo
a mitad de faena. El bloque `gpu_scheduler` de `settings.json` queda explícito
(antes eran defaults implícitos) para poder ajustarlo; con `max_concurrent: 1`
los trabajos van en serie (avisa en vez de encolar).

**3. «Fotos primero (imagen → vídeo)»** en la pestaña Vídeo: casilla + selector de
motor/modelo de imagen (ComfyUI `flux_dev`/`flux_schnell`… o BFL). Por cada escena
se genera su **foto** con su prompt visual y su semilla (`scene_image_params`) y
esa foto entra como primer fotograma del clip **i2v**. Las fotos viven en el
proyecto (`foto_NN.png`) y se reutilizan si ya existen; sin fotos no queda
dependencia del clip anterior. Documentado en `VIDEO-GUION.md` §4.9.

**Estado de la suite en esta sesión:** **796 tests en verde** (2 skips) y
`ruff check .` limpio.

## 0. RESUELTO (30/09/2026): el paralelismo de páginas no compensa

**Decisión:** `ui/document_widget.MAX_PARALLEL_PAGES = 1` y contexto de vuelta a
**chat 8 192 / visión 16 384** en `config/engines.json`. El paralelo se descartó
por medición, no por intuición (el doble de contexto sólo servía para repartir
16 384 entre dos slots de Ollama).

**Medición (auditoría del 30/09 con el `DocumentWorker` real, PDF de 2 páginas,
`qwen3-vl:8b`, modo resumen por página):**

| `parallel` | Corrida 1 | Corrida 2 |
| --- | --- | --- |
| 1 (serie) | 60,1 s | **43,8 s** |
| 2 (lotes) | 59,2 s | **70,1 s** |

→ **1,02x** y **0,62x**: ninguna vez mejor en serie. Y con lotes el widget sólo
recibía **un aviso de progreso por lote** (la barra saltaba de 0 a 2/2), lo que
también quedó corregido: ahora se emite una vez por página desde la propia tarea
(`DocumentWorker._with_progress`).

**Para volver a probarlo** (si algún día hay varios slots de verdad): subir
`MAX_PARALLEL_PAGES`, volver a doblar `num_ctx` (visión 32 768) y cronometrar el
**mismo** documento en las dos configuraciones, mirando segundos por página.

**Estado de la suite en esta sesión:** **596 tests en verde** (2 skips) y árbol
limpio: snapshot del trabajo anterior (`a0e40fb`) + las correcciones de la
auditoría.

**Conectado el `GpuScheduler` (30/09/2026):** las cinco tareas pesadas reservan
GPU con `ui.workers.gpu_slot` — Chat, Documentos, Imagen, Vídeo y Comprensión — y
si no hay hueco piden a los demás motores que suelten su VRAM antes de rendirse
(con el motivo por delante). Los números que lo justifican: Flux Dev pica en
**15 438 MB**, el chat 35b ocupa **~13 GB** y la visión 8b **10 210 MB** de los
16 303 MB de la GPU; con el chat cargado la GPU se queda con **619 MB libres**, el
planificador rechaza el trabajo («quedan 619 MB y se necesitan 1 024 MB») y tras
liberar la VRAM del otro motor hay **14 763 MB** y arranca. Cubierto con
`tests/test_gpu_gating.py` (sin GPU real) y con una prueba en caliente.

**Retención de `output/` hecha (30/09/2026):** `services/output_cleanup.py` con
500 ficheros / 90 días / 8 GB y 24 h de gracia, aplicado al arrancar
(`MainWindow._cleanup_output`, en el pool). Con esos valores **no borra nada** en
el `output/` actual (180 ficheros, 11 días, 110,9 MB), y sólo mira extensiones de
la app (`.png .jpg .jpeg .webp .mp4 .wav .mp3 .srt`): lo que no sea un producto de
Imperator se respeta. Configurable en `settings.json` → `output_limpieza`.

**Arranque arreglado (30/09/2026):** los imports de `kokoro` (9,53 s, arrastra
torch+transformers) y `perception` (8,72 s, torch+transformers) son **perezosos**,
y `is_available()` se resuelve con `find_spec`. Medido: `import ui.main_window`
pasó de **9,85 s a 0,46 s**, el total hasta la ventana de **~11,9 s a ~0,54 s** y
la RAM en reposo de 618 MB a 193 MB (torch/transformers ya no se cargan al
arrancar). Guardado con `tests/test_startup_imports.py`, que importa la app en un
subproceso limpio y falla si vuelve a cargar torch. La primera síntesis con Kokoro
de cada sesión paga su import (~10 s), que antes pagaban todas las sesiones.

**CI reforzado (30/09/2026):** `ruff` 0.16.9 con las reglas de `ruff.toml`
(sintaxis, nombres e imports sin usar y `except` ciego sin justificar; `ruff
check .` pasa limpio y encontró 3 imports muertos al activarlo) y
`constraints-test.txt` con las versiones congeladas de lo que instala el CI, para
que una actualización de PyQt6/numpy no rompa la suite sin tocar el proyecto. Las
reglas de estilo y de orden de imports se dejaron fuera a propósito: sumaban 148
avisos y ninguno era un fallo real.
`pyright` **no** se añadió al CI: necesita Node y en modo `basic` con PyQt6
(`QWidget` sin stubs) da más ruido que señal; sigue disponible como herramienta
local con `pyrightconfig.json`.

**Verificación del resumen global hecha (30/09/2026):** PDF propio de 10 páginas
(16 003 caracteres) con `qwen3-vl:8b` a 16 384 y alcance automático ⇒ global:
**2 bloques → 5 llamadas en 246 s**, entrega de 3 825 caracteres con los 10
epígrafes y las dos partes (ejecutivo + detalle), sin recortar nada
(`truncado: false`) y con la nota de bloques en su sitio.

**Y encontró un fallo real, ya corregido:** el chequeo de cifras de la FASE 4
dependía de cómo se escribieran los millares y de si el resumen abreviaba con
«millones»: marcaba **8 cifras como inventadas** en un resumen fiel. Ahora
normaliza los separadores (espacio / espacio duro / fino / punto / coma) y expande
las abreviaturas de escala (`4,32 millones` = `4 320 000`); medido en el mismo
documento, las cifras inventadas pasan de **8 a 0**. Test-guarda en
`tests/test_fidelidad_cifras.py`. Con esto queda cerrada la lista de mejoras.

**Cómo medir sin engañarse:** el `thinking` del modelo hace que el tiempo varíe
muchísimo entre ejecuciones — las **mismas** 3 páginas en serie tardaron 106 s en
una prueba y 205 s en otra. Hay que comparar **el mismo documento** en las dos
configuraciones y mirar los segundos **por página**, no un total suelto.

**Avisos vigentes tras el cambio:**

- Con dos slots Ollama **reparte el contexto**: si vuelve a aparecer "exceeds the
  available context size" en cualquier motor, hay que doblar su `num_ctx`.
- Los dos modelos no caben juntos en VRAM (8b a 32768 = 9,5 GB; 35b a 16384 =
  12,2 GB). Al cambiar de pestaña Ollama descarga el otro, así que la primera
  página tras el cambio paga la carga del modelo (~5-25 s).
- El chat se queda como estaba (8 192 por petición) gracias a su `num_ctx`
  doblado en `config/engines.json`.

**Pendientes por orden de ganancia medida** (el paralelismo es lo más modesto):

1. Modelo de visión **sin razonamiento** (p. ej. `qwen2.5vl:7b`): quita el ~90%
   de los tokens generados → **5-10x**. Requiere descargar ~5 GB.
2. §1: validar el **ducking** con los MP3 reales de `music/` (sigue pendiente).
3. **Borrador** (speculative decoding) para que la CPU aporte: sin medir.

## 1. Verificar ducking con MP3 reales (≈5 min)

**Estado:** pendiente — el material ya está puesto, falta la escucha de comprobación.
(auditoría del 30/09/2026: `music/` ya no está vacío, tiene 3 MP3)

- El ducking ya está implementado en `core/video_assembler.py` (`_audio_filter_complex`),
  usando `sidechaincompress=threshold=0.05:ratio=8:attack=10:release=250`:
  la voz (`[vsc]`) comprime la música para bajarla mientras hay locución.
- La música de fondo se elige en `ui/assembler_widget.py` (carpeta `music/`, `MUSIC_DIR`).
- **Bloqueo anterior (resuelto):** `music/` estaba vacío (0 archivos). Hoy tiene
  `Deep Forest - Freedom Cry.mp3`, `Qué soy yo para ti.mp3` y `Say it right.mp3`,
  así que el pendiente es solo escuchar el resultado y decidir.

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

### 5.8 Procesado por lotes: activado tras forzar los slots en Ollama

Idea: solapar páginas aprovechando que Ollama atiende varias peticiones a la vez.
Se implementó en `DocumentWorker` (`parallel=`, lotes con `_chunks`), dejando la
**lectura** de cada página y el montaje del resultado en el hilo del worker
(PyMuPDF no es thread-safe) y conservando orden, progreso y cancelación. Medido
con `qwen3-vl:8b`:

| caso | tiempo | ganancia |
| --- | --- | --- |
| 2 peticiones crudas por HTTP, con dos slots activos | 140 s serie / 83 s paralelo | **1,68x** |
| `DocumentWorker` completo, 3 páginas, carga con **un** slot | 106 s serie / 126 s paralelo | **0,84x** |
| `DocumentWorker` completo, 3 páginas, con `OLLAMA_NUM_PARALLEL=2` y `num_ctx=32768` | 205 s serie / 155 s paralelo | **1,32x** |

La diferencia enseña lo importante: **Ollama decide cuántos slots abre al cargar
el modelo** y el cliente no lo controla. Con un solo slot las peticiones se
encolan y el paralelismo solo añade sobrecarga y contención de caché; con dos, se
solapan de verdad (1,68x).

Decisión final: **activado** (`MAX_PARALLEL_PAGES = 2`) con estos ajustes:

- `OLLAMA_NUM_PARALLEL=2` como variable de entorno del usuario y **reinicio de
  Ollama** (la variable decide cuántos slots abre al cargar: uno solo no
  paraleliza nada). Verificado: `setx` + reinicio y el servidor vuelve en ~10 s.
- `ollama_vision.parametros.num_ctx: 32768` (16 384 para cada página), porque
  Ollama **reparte el contexto entre los slots**.
- `ollama.parametros.num_ctx: 16384` para el **chat**: con dos slots le tocan
  8192 por petición, los mismos que tenía antes del cambio. Medido: el 35b ocupa
  12,2 GB de VRAM con 16384, así que cabe y no empeora el chat. Sin este ajuste,
  el chat se habría quedado con 4096 por petición.
- `config/engines.json` y `config/settings.json` son locales (no versionados); el
  único cambio de código es `MAX_PARALLEL_PAGES` y el `num_ctx` por defecto.

**Aviso sobre las medidas:** el razonamiento del modelo hace que el tiempo
absoluto varíe mucho entre ejecuciones (las mismas 3 páginas en serie tardaron
106 s en una prueba y 205 s en otra). Por eso la ganancia de 1,32x se refiere a
esa comparación concreta, y con documentos de más páginas debería acercarse al
1,68x de las peticiones crudas (con 3 páginas, el segundo lote va cojo: solo una
página). Conviene medirlo con el documento real.

6 tests cubren lotes de dos en vuelo, lotes de uno, orden de las páginas,
resultado correcto por página y cancelación entre lotes.

## 6. Pipeline de documentos por fases (2026-09-28)

Documentos deja de ser un `if` por modo y pasa a ser el pipeline del diagrama:

```
PDF
 ↓
[FASE 0: Detección] ─ ¿el usuario pide OCR? ─┬─ sí → FASE 1 OCR (VL)
                                             └─ no → texto embebido
 ↓
¿el usuario pide tablas? ─┬─ sí → FASE 1 Tablas (VL)
                          └─ no → saltar
 ↓
[FASE 2: Normalización] ─ ¿el usuario pide resumen?
                            ├─ global (documento completo)
                            └─ por apartados (por página)
 ↓
[FASE 4: Validación ligera]
 ↓
[FASE 5: Entrega]
```

**`services/document_pipeline.py` (nuevo, sin Qt) es el plan**; `DocumentWorker`
es la ejecución. El módulo decide y el worker pregunta:

- `PipelinePlan` (frozen) reúne las decisiones: `route(has_text)` resuelve la
  FASE 0 (`texto` u `OCR`), `phases()` lista las fases que se van a ejecutar,
  `tables_extra()` dice si las tablas van como bloque aparte y los prompts se
  piden aquí (`image_prompt`, `text_prompt`, `global_prompt`, `ocr_prompt`),
  delegando en `document_params` para no duplicar reglas.
- `normalize_pages` (FASE 2) une las páginas y devuelve si hubo recorte;
  `validate` (FASE 4) produce los avisos —documento recortado, entrega vacía—
  sin bloquear nada; `deliver` (FASE 5) es el único sitio que monta el texto.

**Lo que gana la pestaña Documentos:**

- **Casilla «Forzar OCR (FASE 1)»**: transcribe con el modelo de visión aunque
  el PDF traiga capa de texto (útil cuando ese texto es malo). Apagada
  (por defecto) todo sigue igual: si la página tiene texto se usa tal cual y
  solo van por OCR las que no lo tienen.
- **Casilla «Extraer tablas además (FASE 1)»**: añade un bloque
  `--- Tablas (Página N) ---` con el resultado principal (transcripción o
  resumen), reutilizando el texto ya extraído en vez de volver a rasterizar.
  En el modo Extraer tablas se desactiva: allí ya son el cuerpo.
- Las dos casillas se persisten (`document_force_ocr`, `document_tables`) y se
  bloquean mientras hay trabajo, como el resto de controles.
- Los modos siguen siendo los de siempre y **el formato del resultado no
  cambia** (`--- Página N (texto|OCR) ---` por página; en global, un único
  resumen): lo que cambia es que ahora cada fase tiene su método y su plan.

**Validación ligera (FASE 4), con aviso pero sin bloquear:** si el documento se
recortó por el presupuesto de caracteres se añade `TRUNCATED_NOTICE`; si la
entrega se queda sin nada de texto, `EMPTY_RESULT_NOTICE` (el modelo no
respondió) o `NO_CONTENT_NOTICE` (las páginas no dieron contenido).

**Verificado:** 38 tests nuevos en `tests/test_document_pipeline.py` (plan,
rutas, prompts, normalización, validación, entrega y la ejecución por el
worker: OCR forzado, tablas junto al resumen y junto a la transcripción, y los
avisos).

### 6.1 Un resumen de 19 páginas tardaba 20 minutos: eran 19 llamadas al modelo

Reportado con un documento real: **19 páginas de texto corto, 20 minutos**.

**Medido** (`tools/bench_resumen.py`, página de ~900 caracteres, `qwen3-vl:8b`,
`OLLAMA_NUM_PARALLEL=2`, así que el contexto útil por petición es la mitad del
`num_ctx`):

| caso | s/página | 19 páginas |
| --- | --- | --- |
| visión 8b, ctx 16384 (8192/slot) | 64,4 | 20,4 min |
| visión 8b, ctx 32768 (16384/slot) — la config actual | 49,9 | 15,8 min |
| visión 8b, ctx 65536 (32768/slot) | 143,3 | 45,4 min |
| texto 9b / texto 35b, ctx 16384 | 96,5 / 67,4 | 30,6 / 21,3 min |
| **visión 8b, ctx 32768, resumen GLOBAL** | — | **0,8 min (47,7 s)** |

No era un atasco: era **una llamada por página** con un modelo que razona. Por
página el modelo genera **5 888 tokens** para devolver ~650 caracteres (23 248
caracteres de `thinking`): el razonamiento se lleva ~90 % del tiempo. 19 × ~50 s
= 15-20 min. Tres detalles que confirman el diagnóstico:

- subir el contexto a 65536 **empeora** (143 s/página): el modelo llena con
  razonamiento la ventana que le des;
- un modelo de **texto** (9b) es *peor* aquí (96,5 s): el razonamiento no viene
  de la visión;
- el paralelismo por lotes no rescata el caso: con dos slots la GPU se reparte y
  dos páginas a la vez cuestan ~2× cada una.

**Arreglo (elegido con los números delante): resumen global automático.** El
alcance deja de ser una casilla y pasa a **selector de tres opciones**:

- **Automático** (por defecto): hasta 5 páginas, resumen página a página; a
  partir de **6** (`document_params.AUTO_GLOBAL_MIN_PAGES`), **una sola llamada**
  con el documento completo.
- **Global**: lo fuerza siempre.
- **Por página**: mantiene el resumen por página a mano (con su coste).

La preferencia `document_global` era booleana («1» global / «0» por página) y
`normalize_scope` sigue leyendo esos valores, así que no se pierde nada al
actualizar. El alcance se resuelve en `PipelinePlan.with_page_count()` al abrir
el PDF (FASE 2) y queda en el log («Documentos: 19 páginas, resumen global → …»).

**Verificado de punta a punta** (`tools/verify_alcance_documentos.py`: PDF real
de 19 páginas, motor real y `DocumentWorker`): **1 llamada** al modelo, **30,6 s**
en total, alcance resuelto `global` y el resumen con los cinco apartados.

### 6.2 El resumen rápido tenía errores: recorte, prompt y idioma

Reportado justo después de activar el alcance automático: «ahora es más rápido
pero tiene errores el resumen: errores de contenido, frases mezcladas,
omisiones y algunas interpretaciones incorrectas». Reproducido con
`tools/verify_calidad_resumen.py` (PDF de 19 páginas con 76 apartados y cifras
verificables) y **tres causas medidas**, ninguna intuida:

1. **Recorte silencioso.** `MAX_GLOBAL_CHARS = 12 000` con un documento de
   14 285: entraba el 84 % y las páginas 14-19 se quedaban fuera → **40 cifras
   omitidas**.
2. **El modelo extrapolaba lo que no veía.** Las 3 «cifras inventadas» del
   primer informe (`221329`, `11795`, `017`) eran de la **página 17**, que no se
   le había enviado: el modelo **dedujo la serie aritmética** que veía en las
   páginas 1-13 y se inventó los valores que faltaban. Nunca inventó un dato
   suelto: completó un patrón. Es la peor forma del fallo, porque el resultado
   parece coherente.
3. **El prompt pedía «5-10 frases».** Para 76 apartados eso obliga a mezclar
   apartados en una misma frase y a resumir con rangos construidos a ojo.
4. **Idioma: el modelo traducía.** Con `document_language: "auto"` el resumen
   salía en el idioma de las **instrucciones**, no en el del documento: un
   documento real en euskera daba el cuerpo en castellano con los títulos en
   euskera (y, en las pruebas, el mismo documento en castellano salía a veces en
   inglés). Pedir «responde en el mismo idioma» no bastaba: apareció en la
   medición nº 3, no en las dos anteriores, y con el documento del usuario —que
   no está en las pruebas— salió igual.

**Cambios:**

- `MAX_GLOBAL_CHARS` pasa a ser el presupuesto **por llamada** (16 000) y un
  documento que no cabe **ya no se recorta**: se resume por bloques
  (`document_pipeline.summary_blocks`, corta por párrafos y por frases
  completas) y después se resume el conjunto. Con 19 páginas son 3-4 llamadas
  en vez de 19, y **ninguna página se pierde** (`BLOCKS_NOTICE` lo indica).
- **Prompts reescritos**: fuera el «5-10 sentences»; un bloque por apartado con
  su dato clave, 1-2 frases, y —al ver ya el texto completo— permitido agrupar
  apartados que repiten estructura y dar el rango **cuyos dos extremos están en
  el texto**. La prohibición de extrapolar sigue: «never quote a figure that is
  not literally in the text… never complete the values of items the document
  does not state».
- **Idioma resuelto por detección** (no por confianza en el modelo):
  `document_params.detect_language` identifica el idioma del documento por
  palabras funcionales —comparadas por **prefijo**, que es como funciona el
  euskera (`lan`, `lanaldia`, `laneko`)— y el worker fija ese idioma en el plan
  (`PipelinePlan.with_language`) cuando el alcance es «auto»: el prompt lo pide
  **por su nombre** (`Write the answer in Basque (euskera)…`) y no con un
  «responde en el mismo idioma» que el modelo ignoraba. Si la detección no está
  clara, se queda en «auto». Además la instrucción va antes de las reglas de
  estilo y un idioma elegido a mano **nunca** se pisa.
- **FASE 4 con verificación de cifras** (`document_pipeline.check_fidelity`):
  avisa de las cifras del resultado que no están en el documento (el error
  real). De las **omitidas** no avisa a propósito —un resumen es selectivo y
  avisar siempre acaba ignorándose—, pero el dato se sigue midiendo con el
  verificador.

**Medido (mismo documento, `qwen3-vl:8b`, 1 llamada):**

| | por página | global con recorte | global ahora |
| --- | --- | --- | --- |
| tiempo | ~16-20 min | 51,8 s | **67,6 s** |
| cifras inventadas | 0 | 3 (extrapoladas) | **0** |
| cobertura | 19/19 páginas | 13/19 páginas | **19/19** |
| idioma | castellano | castellano/inglés | **castellano** |
| longitud | — | 1 306 chars mezclando temas | **696 chars, 4 bloques** |

Mismo banco con el documento **en euskera** (`--eu`, 13 401 caracteres, 102,8 s):
``documento=eu resumen=eu``, 0 cifras inventadas, 609 caracteres.

Las 51 cifras «omitidas» del último informe son los valores intermedios de las
series, que la agrupación resume con sus extremos reales: no son pérdidas.

**Pendiente:** un modelo de visión **sin razonamiento** (`qwen2.5vl:7b`) bajaría
los ~67 s a segundos. Y el informe de calidad (`tools/verify_calidad_resumen.py`)
queda como banco para comparar cada cambio de prompt.

### 6.3 Corrección de un resumen real: cuatro fallos, cuatro reglas

El resumen de un **convenio laboral real** (en euskera) salió «bastante bien
pero con errores importantes», corregido a mano apartado a apartado. Los fallos
no eran aleatorios: eran cuatro patrones, todos del mismo tipo —el modelo
**movía** información de sitio—.

| fallo | patrón |
| --- | --- |
| «24 **días** de trabajo» cuando el documento dice **24 horas** | cifra con la unidad cambiada |
| «no supera 24 horas en dos semanas» (no está en el documento) | cifra traída de otro apartado |
| el trabajo nocturno dentro de *Ordutegia*; el descanso semanal dentro del descanso entre jornadas | información movida de apartado |
| «excepto en trabajos nocturnos» cuando el documento dice que los nocturnos **no** pueden hacer horas extra | **sentido invertido** |
| de 9 permisos retribuidos quedaron 2 | listas comprimidas |

La causa de los dos últimos era **nuestra**: al pedir «un bloque de 1-2 frases
por apartado» y permitir agrupar apartados, el modelo comprimía y mezclaba.
Cuatro reglas nuevas en `SUMMARY_STYLE` (y listas enteras también en
`GLOBAL_PROMPT`/`CHUNK_PROMPT`):

1. `do not turn hours into days`, `do not reuse a figure taken from a different
   section` → cada cifra con su concepto y su unidad;
2. `One section, one block` + `Never move a statement` + `never merge two
   different sections` → nada cambia de apartado;
3. `keep all of them` / `a list is not summarised by dropping items` → las
   listas se conservan enteras;
4. `state exactly who is forbidden and who is exempt` / `never invert the sense
   of the rule` → las prohibiciones, con sus excepciones y en su sentido.

`tests/test_document_pipeline.py::ReglasDeEstiloTest` (5 tests) fija las cuatro
reglas y que los prompts globales protejan las listas.

**Medido con `--convenio`** (convenio sintético de una página con los cuatro
patrones, `qwen3-vl:8b`): 9 apartados, cada uno con su bloque, **1 712
caracteres**, `documento=eu resumen=eu`, **0 cifras inventadas y 0 omitidas**.
Los cuatro fallos ya no aparecen: dice «24 ordu», mantiene «debekatuta daude
gaueko langileentzat, indar majarrak izan ezik» (prohibidos con su excepción, en
el sentido correcto) y lista los seis permisos completos.

**El compromiso, con números.** Fidelidad y brevedad tiran en direcciones
opuestas y ahora gana la fidelidad:

| documento | resumen con el prompt compacto (el que fallaba) | ahora |
| --- | --- | --- |
| convenio, 1 801 caracteres | 607 (omitiendo la mitad) | **1 712 (95 %)** |
| 19 actas repetidas, 14 190 caracteres | 696 (4,9 %, agrupando) | 13 393 (94 %, enumerando) |

Es decir: en documentos con muchas repeticiones el resumen deja de ser breve
porque conserva todos los ítems. Si se prefiere breve, la palanca es una sola
frase del prompt (`keep all of them…`, en `SUMMARY_STYLE`): quitarla vuelve al
resumen corto con el riesgo de mezclar apartados y perder ítems.
`tools/verify_calidad_resumen.py` mide las dos cosas: `--convenio` para el caso
con listas y sin él para el de series repetidas.

### 6.4 Resumen ejecutivo: el 25 % no siempre es posible

Elegido el resumen **ejecutivo** (20-30 % del documento, aceptando perder
detalle secundario), se implementó como estilo por defecto (`STYLE_EXECUTIVE`)
con el detallado a mano, y se midió iterando el prompt:

| versión | qué se le pide | resultado (convenio de 1 801 caracteres) |
| --- | --- | --- |
| «un quinto del documento» | longitud relativa | **83 %** — el modelo no la mide |
| + tope explícito `at most N palabras` (`max_words_for`, 25 %) | longitud calculada del documento | **54 %** (981 caracteres, 1 llamada, 0 inventadas, 0 omitidas) |
| + segundo pase de compresión (`EXECUTIVE_COMPRESS`) | comprimir el resumen a N palabras | **65 %** y el doble de tiempo → **apagado por defecto** |

**El hallazgo, que es lo importante:** el tope de palabras sí funciona (83 % →
54 %), pero el segundo pase **empeora** porque el modelo vuelve a «conservarlo
todo». Y la razón de fondo es que el convenio es una **lista de ~40 datos en
1 801 caracteres**: un resumen de 450 no puede contener 40 datos. En un
documento denso, bajar del ~50 % exige **omitir datos**, y eso no lo arregla
ningún prompt: es una decisión de producto (qué se omite), no de ingeniería.

Lo que sí está garantizado y medido en el ejecutivo: idioma del documento,
0 cifras inventadas, 0 cifras omitidas y las listas completas, con **una sola
llamada** (~110 s por documento).

Pendiente de decidir con el usuario (dos vías reales al 20-30 %):

1. **Dos secciones** en la misma entrega: resumen ejecutivo breve + detalle por
   apartado debajo (dos llamadas más, es la única forma de tener las dos cosas).
2. **Omitir de verdad**: recortar el resumen por su cuenta (p. ej. dejar la
   primera frase de cada bloque), aceptando perder datos secundarios.

### 6.5 Entrega en dos partes: implementada, medida y **el modelo no la sostiene**

Elegida la vía «resumen ejecutivo breve arriba + detalle por apartado debajo»
(es la única forma de tener las dos cosas), se implementó como estilo por
defecto (`STYLE_TWO_PARTS`): dos llamadas sobre el mismo texto, con los
encabezados `== Resumen ejecutivo ==` y `== Detalle por apartados ==`.

**Medido** (`--convenio`, `qwen3-vl:8b`, 1 801 caracteres de documento):

| parte | caracteres | respecto al documento |
| --- | --- | --- |
| resumen ejecutivo | 1 371 | **76 %** (no es un resumen) |
| detalle por apartados | 1 806 | 100 % |
| entrega total | 3 177 | 176 % |

Y 3 llamadas / 546 s. Dos conclusiones medidas:

1. **El tope de palabras no se respeta de forma fiable.** El mismo
   `at most 65 words` dio 981 caracteres (54 %) en una ejecución y 1 371 (76 %)
   en otra: con un prompt largo y reglas de fidelidad, el modelo prioriza «no
   perder datos» sobre el límite. El ejecutivo oscila entre el 54 % y el 76 %.
2. **La parte «breve» acaba repitiendo el detalle**, así que la entrega ocupa
   casi el doble que el documento sin añadir información.

**Conclusión honesta:** en un documento cuyo contenido es una lista de ~40 datos
(el caso del convenio), **un resumen fiel del 20-30 % no es alcanzable con este
modelo**. No es un problema de prompt: 40 datos no caben en 450 caracteres. Lo
único que queda garantizado, y está medido, es el **detalle por apartado fiel**
(0 cifras inventadas, 0 omitidas, listas completas, idioma del documento) y la
opción de dos partes con su ejecutivo *cuando el modelo colabora*.

Palancas pendientes, por orden de eficacia esperada:

1. **Modelo de visión sin razonamiento** (`qwen2.5vl:7b`), que ya está descartado
   para otra cosa: aquí ayudaría a que el resumen ejecutivo sea estable.
2. **Recorte mecánico del ejecutivo** (quedarse con la primera frase de cada
   bloque) en vez de pedírselo al modelo: predecible, pero pierde datos.
3. **Volver al detalle por apartado como único estilo** si no se quiere leer dos
   veces lo mismo.

Suite completa: **422 tests en verde** (1 skip).

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

## 7. Sesión (2026-09-30): planificador de GPU, orquestador, percepción y comprensión

Se evaluó una propuesta de arquitectura externa (pipeline de entrada, T5 para
limpiar OCR, `*.plan` de TensorRT para todo, montaje de vídeo, capa de GPU) y se
implementó **solo la parte que era mejora real**. Suite: **573 tests en verde**.

**Lo que se descartó y por qué** (queda escrito para no repetir el análisis):

- **T5 para limpiar texto OCR: descartado.** `t5-small/base` está preentrenado
  esencialmente en inglés; para castellano/euskera degrada. Y el cuello real de
  Documentos no es el ruido del texto, sino el `thinking` del modelo de visión y
  el contexto (§6).
- **Capa de entrada por tipo de contenido: ya existía** (las pestañas).
- **Detección PDF texto/escaneado: ya existía** (`services/document_pipeline.py`).
- **Montaje de vídeo: ya existía** (`core/video_assembler.py` + Ensamblador).
- **`*.plan` governando Ollama y ComfyUI: imposible.** Son runtimes externos
  (llama.cpp/CUDA y PyTorch/CUDA). Los motores compilados solo aplican a
  inferencias propias (YOLO/CLIP/OCR), no a los motores HTTP.
- **«Fotograma cada N ms»: inviable.** Se muestrea cada N **segundos**, en el
  centro de cada intervalo.

**Fase 1 — `core/gpu_scheduler.py`** (`tests/test_gpu_scheduler.py`, 46 tests):

- Tres prioridades (`perception`/`interactive`/`batch`), concurrencia máxima,
  reserva de VRAM (`vram_headroom_mb`) y aplazamiento de lotes cuando la GPU está
  ocupada. VRAM medida con `nvidia-smi` (`--query-gpu=...,--format=csv`).
- **Falla abierto**: sin `nvidia-smi` (o desactivado) nunca bloquea. El sondeo va
  fuera del lock y los tests inyectan un `GpuProbe` falso (no necesitan GPU).
- `settings.json` → `gpu_scheduler` (nuevo esquema opcional). Menú
  *Motores → Estado de la GPU…*, con el sondeo en el pool (no en el hilo de UI).

**Fase 2 — router de intención y orquestador** (`tests/test_intent_router.py`):

- `services/intent_router.py`: reglas léxicas es/eu/en, sin LLM. Umbral 2.5 para
  exigir señal compuesta («pdf» no enruta; «resume este pdf» sí).
- `services/task_orchestrator.py`: resuelve la intención contra las tareas
  registradas. `TaskDeps` gana `orchestrator` (con valor por defecto, para no
  romper a quien lo construya a mano); `TaskManager` lo crea y el chat recibe una
  factory propia (`_chat_factory`).
- El chat **emite** `routed` y `MainWindow` ofrece abrir la pestaña. El chat no
  abre pestañas ni conoce widgets.

**Fase 3 — percepción estructurada** (`tests/test_perception_engine.py`):

- `engines/base_engine.py`: nueva familia `PerceptionEngine` + `Perception`
  (inmutable, `merge()`/`summary()`/`to_dict()`).
- `engines/perception.py`: `yolo` (detección), `clip` (etiquetado zero-shot) y
  `paddle_ocr` (OCR local), con import protegido y runner/modelo inyectables.
  Registrados en `ENGINE_FACTORY`, `DEFAULT_ENGINES`, `engines.example.json` y
  las plantillas de la UI de motores. **Deshabilitados por defecto**.
- `parse_ocr_output` tolera las dos API de PaddleOCR (2.x anidado por página y
  3.x con `rec_texts`); el test lo cubre con dobles.
- Avisos honestos: PaddleOCR no tiene modelo de euskera (se usa `latin`) y CLIP
  no describe: puntúa un vocabulario cerrado.

**Fase 4 — comprensión de vídeo** (`tests/test_video_understanding.py`):

- `services/video_understanding.py` (puro): `sample_timestamps` (centro de cada
  intervalo, tope de fotogramas, `_spread` uniforme si se pasa), `srt_to_text`,
  `aggregate_prompt` y `build_report`.
- `core/frame_extractor.py`: fotogramas PNG y audio WAV 16 kHz mono con el ffmpeg
  de `imageio-ffmpeg`, con `timeout` y comprobación de salida.
- `ui/understanding_widget.py` + `tasks/understanding.py`: pestaña **Comprensión**
  (7ª). El worker ejecuta el pipeline por fases, respeta el planificador de GPU
  (`PERCEPTION_WAIT_S`) y **avisa** cuando falta un motor en vez de inventar.

**Pendiente de esta línea de trabajo:**

1. Medir con un vídeo real: coste por fotograma con YOLO en GPU y cuántos
   fotogramas hacen falta para que el resumen mejore (ahora 24 por defecto).
2. Export a TensorRT (`.engine`/`.plan`) de YOLO y, si compensa, runner de ONNX
   para PaddleOCR: hoy se acepta la ruta del motor pero **no se compila nada**.
3. Ejecución automática de la intención (hoy `MainWindow` pregunta): el punto de
   extensión es `services/task_orchestrator.py`.

## 8. Sesión (2026-09-30, segunda): auditoría externa y cierre de hallazgos

Auditoría independiente de todo el árbol (código, configuración, tests, git y
seguridad); el informe completo está en `AUDITORIA-2026-09-30.md`. Se cerraron
**todos los hallazgos de severidad media y baja**; el único alto (la clave real
de BFL en `.env` en claro) queda en manos del autor, con la recomendación de
rotarla si hubo cualquier duda de exposición y de verificar que la carpeta no
está sincronizada en la nube.

**Lo que se verificó (no solo se afirmó):** `ruff check .` limpio, `compileall`
OK, árbol de git limpio, `.env` fuera del historial (`git log --all -- .env`
vacío), **ningún** `shell=True`/`os.system`/`eval`/`exec`/`pickle`, **ninguna**
llamada `requests` sin `timeout` y todas las descargas acotadas por
`MAX_DOWNLOAD_BYTES`.

**Correcciones aplicadas:**

- **Lockfile de runtime.** `constraints.txt` (el `pip freeze` del venv de
  referencia, 204 paquetes) y el README explica el
  `pip install -r requirements.txt -c constraints.txt`. El agujero real: los
  tests estaban congelados, pero `torch`/`onnxruntime`/`ctranslate2` (que
  arrastran Kokoro, Piper y Whisper) no lo estaban.
- **62 tests nuevos** (631 → 693) en los módulos que más efectos secundarios
  tienen y menos red tenían: `piper_tts`, `kokoro_tts`, `whisper_engine`,
  `frame_extractor`, el formulario de motores y las cinco pestañas sin test
  propio (Vídeo, Imagen, Audio, Ensamblador, Comprensión). Todos con dobles: ni
  motores, ni GPU, ni red, ni ffmpeg.
- **Estado compartido entre hilos.** `PiperTTSEngine._loaded` y
  `OllamaEngine._last_used_model` van bajo `threading.Lock`: son las dos únicas
  piezas que escribían desde un worker y se leían desde otro hilo (la liberación
  manual de VRAM lee el segundo).
- **Saneado de entrada.** El nombre de voz de Piper se valida contra
  `list_voices()` antes de construir la ruta: un valor editado a mano en
  `settings.json` ya no puede sacar la lectura de `voices/`.
- **Documentación sincronizada.** `ARCHITECTURE.md` §8 decía
  `MAX_PARALLEL_PAGES = 2` y contextos 32768/16384 (y se contradecía más abajo);
  ahora refleja el estado real (1 y 8192/16384) y el nº de tests. `TODO` §1 ya no
  da `music/` por vacío (hay 3 MP3: el pendiente es escuchar el ducking) y
  `config/engines.json` recupera `musicgen`, que solo estaba en los defaults y en
  la plantilla.
- **Higiene.** `*.lnk` fuera de git (es un binario con rutas de la máquina y el
  README ya explica cómo regenerarlo), `__pycache__/` de la raíz borrado
  (`_min_test`/`_min_test2` sin `.py`) y el pin de `ruff` en un solo sitio
  (`constraints-test.txt`), no duplicado en el workflow.
- **Herramientas de desarrollo.** `requirements-dev.txt` con `ruff` y `pyright`
  (documentado en el README; `pyright` sigue fuera del CI por el Node y el ruido
  de PyQt6, pero ahora se instala con un comando).
- **Log a fichero activado.** `settings.json` → `rutas.logs = "logs"`: los fallos
  intermitentes de motores dejan traza en `logs/plataformaia.log` (rotado a
  diario) en vez de desaparecer con la consola.

**Pendiente por decisión del autor:** rotar/verificar la credencial `BFL_API_KEY`
de `.env` (hallazgo 3.1 del informe) y, opcionalmente, moverla a Credential
Manager; `ConfigManager.resolve_api_key` es el punto único donde cambiarlo.

## 9. Sesión (2026-10-02): el vídeo por escenas petaba por memoria

**Síntoma:** generar un guion de varias escenas moría tras la primera. Evidencia
en disco: `projects/el-secreto-del-trigo-dorado/` tenía 6 escenas en el guion
pero **solo `escena_00.mp4`** (más el `ultimo_00.png` de encadenado). Al reintentar
se rehacía desde la escena 1 y volvía a caer en el mismo punto, porque no había
reanudación.

**Diagnóstico:** ComfyUI ya libera su VRAM tras cada clip (`POST /free` dentro de
`generate_video`), pero **el proceso de Python no devolvía al SO la memoria que
dejaba de usar** (MP4 descargado de cada escena + arenas de `pymalloc`). En una
secuencia larga el proceso crecía escena a escena hasta que Windows lo mataba.

**Arreglo (3 piezas):**

- `core/memory.py` (nuevo): `release_memory()` = `gc.collect()` + recorte del
  *working set* en Windows (`EmptyWorkingSet` / `SetProcessWorkingSetSize(-1,-1)`).
  Best-effort, nunca lanza y con la parte del SO inyectable para tests
  (`tests/test_memory.py`).
- **Reanudación por escena.** `Project.base_seed` y `find_project()` en
  `services/video_project.py`; `Project.resume_index()` salta el **prefijo
  contiguo** ya generado (un hueco invalida lo posterior por el encadenado i2v).
  `SequenceWorker` continúa desde la primera que falta, reconstruye el i2v desde
  el último fotograma guardado y reutiliza la **misma semilla común** para no
  perder la coherencia. Casilla «Reanudar proyecto» (encendida por defecto).
- **Liberación entre escenas.** Tras guardar cada `escena_NN.mp4`, el worker pide
  al motor que suelte su VRAM, borra el fotograma de encadenado ya consumido y
  llama a `release_memory()`. Así el proceso no acumula nada de una escena a la
  siguiente.
- **Bonus de coherencia de configuración:** `scene_generate_params(width, height)`
  ahora respeta el preset/tamaño elegidos en la pestaña (antes la generación por
  escenas usaba solo el preset por defecto del formato e ignoraba esos controles).

**Resuelto (04/10/2026):** probado en caliente — ver §16. La secuencia del
03/10 completó 1/6 → 6/6 y dejó `secuencia.mp4` **sin cortes**, así que la
liberación de memoria entre escenas aguantó una corrida larga real.

## 10. Auditoría (2026-10-03): por qué el vídeo por escenas solo dejaba la escena 1

**Síntoma reportado:** «con solo texto no puedo hacer el vídeo». Los dos
proyectos de `projects/` tienen el guion completo (**6 escenas**) pero **solo
`escena_00.mp4`** (+ `ultimo_00.png`, el fotograma de encadenado): la generación
se corta después de la primera escena.

**Lo que se verificó (medido, no asumido):**

- Los servicios externos están vivos y son los correctos: **ComfyUI** en
  `127.0.0.1:8188` y **Ollama** en `127.0.0.1:11434` a la escucha; los cuatro
  workflows de imagen y los dos de Wan (`wan22_t2v_workflow_api.json`,
  `wan22_i2v_workflow_api.json`) existen en `C:\CComfyUI` y tienen los nodos que
  el motor parchea (`6`/`7` `CLIPTextEncode`, `KSamplerAdvanced`, latent
  `EmptyHunyuanLatentVideo`/`WanImageToVideo`, `LoadImage`).
- El motor **sí genera**: `plataformaia.log` tiene **28 líneas `WanVideo`** con
  clips T2V e I2V terminados (`/history … completed=True`, MP4 guardado).
- El log **no tiene ni un `ERROR`, ni un `Traceback`, ni una línea `escena`**
  (comprobado por conteo). El corte tras la escena 1 **nunca se registró**.

**Causa raíz del bloqueo de diagnóstico (lo que hay que arreglar):**
[`ui/workers.py`](../../../Imperator/ui/workers.py:49) → `ThreadWorker.run` **no
registraba los `EngineError`** (el fallo más habitual: motor caído, OOM, timeout,
workflow inválido) y bajaba las excepciones inesperadas a `logger.debug`,
**invisible con `log_level=INFO`** (el valor por defecto). `TaskWidget._on_failed`
solo escribe en la barra de estado. Resultado: la razón del corte no quedaba en
`logs/plataformaia.log`, así que no había forma de saber qué pasaba.

**Arreglo aplicado:**

- `ThreadWorker.run`: los `EngineError` reales se registran a **ERROR**; las
  excepciones inesperadas con `logger.exception` (con traza). La **cancelación**
  del usuario sigue sin registrarse (no es un fallo).
- `SequenceWorker.work`: una línea de `INFO` por escena
  («Proyecto 'x': generando escena N/M (modo t2v|i2v)») antes de cada
  `generate_video`, para saber en qué escena se quedó si vuelve a cortarse.

**Aclaración de uso (no es fallo de código):** el modo del widget se persiste en
`settings.json` → `last_models.video` y hoy está en **`i2v`** (imagen→vídeo).
Con ese modo, el botón **Generar** exige una imagen y **rechaza el texto solo**:
de ahí el «con solo texto no puedo». Para texto→vídeo hay que elegir
**Modo → Texto a vídeo (t2v)**. Y el vídeo final con voz/subtítulos/música no lo
entrega la pestaña Vídeo (que produce los clips y `secuencia.mp4`) sino
**Audio + Ensamblador**.

**Verificación:** `ruff check` limpio y **70 tests de vídeo en verde**
(`test_video_sequence`, `test_video_pipeline`, `test_video_script`).

**Resuelto (04/10/2026):** no queda nada que reanudar: `projects/` solo tiene
`el-secreto-del-trigo-dorado/` y **vacía** (sin clips ni `secuencia.mp4`), y el
proyecto `el-secreto-del-trigo-de-basajaun` no existe. El log del 03/10 ya
muestra la secuencia completa (1/6 → 6/6), así que el diagnóstico del corte
quedó cerrado — ver §16. La ruta de reanudación sigue cubierta por
`tests/test_video_pipeline.py` y `tests/test_video_sequence.py`.

### 10.1 El guion no se generaba: el timeout del chat era de 120 s en duro

**Síntoma:** «no se pudo generar el guion. Ollama tardó demasiado en responder
(timeout)». El log del servidor lo deja claro: `qwen3.6:35b` **estaba
respondiendo** (99-108 t/s) y llevaba **6709 tokens generados** cuando el cliente
cortó la petición a los **2m0s** (`[GIN] ... 500 | 2m0s | POST /api/chat`).

**Causa:** [`engines/ollama.py`](../../../Imperator/engines/ollama.py:63) fijaba
`self._chat_timeout = 120.0` **en duro**. El guion de escenas es una tarea de
razonamiento (prompt de sistema largo + esquema JSON) y con un modelo que
"piensa" (qwen3.x) los 120 s se agotan a mitad; es el mismo caso que ya se
resolvió en la visión con 600 s configurables.

**Arreglo:**

- `OllamaEngine` gana `_CHAT_TIMEOUT = 600.0` y un parámetro `timeout`
  configurable por motor en `engines.json` (`ollama.parametros.timeout`),
  cableado en `EngineManager._build` igual que el de la visión. La plantilla
  `engines.example.json` y la config local quedan con `"timeout": 600`.
- El mensaje de timeout ahora dice el valor usado y cómo subirlo.
- 3 tests nuevos (`test_ollama_engine`, `test_ollama_defaults`).

**Contrapartida anotada:** la petición es bloqueante, así que durante ese margen
la cancelación no la aborta; se resolvería con **streaming** de `/api/chat` (que
además daría progreso real), pendiente.

**Si el guion sale truncado** (no un timeout, sino `done_reason: length`), el
siguiente ajuste es subir `ollama.parametros.num_ctx` (hoy 8192 en la config
local; 16384 ya funciona con el 35b según §5.8). Se vería en el log como
«Ollama: respuesta cortada por el límite de tokens».

**Verificación:** `ruff check .` limpio, config JSON válida y **58 tests en
verde** (motor Ollama, defaults, gestor de motores y secuencia de vídeo).

### 10.2 El guion no parseaba: JSON cortado por el contexto (``num_ctx``), no por tiempo

**Síntoma:** «no se pudo interpretar el guion del modelo: El JSON del guion no se
pudo parsear: Expecting ',' delimiter: line 67 column 6 (char 4826)». En el log,
justo antes, la pista que lo explica:

```
2026-10-03 07:01:52 WARNING [engines.ollama] Ollama: respuesta cortada por el límite de tokens
```

**Causa (medida):** con `OLLAMA_NUM_PARALLEL=2` en el entorno y
`ollama.parametros.num_ctx: 8192`, Ollama sirvió **`context_length: 8192`**
(comprobado con `/api/ps`). `qwen3.6:35b` **razona** y llena la ventana con
`thinking` antes de escribir el JSON: el guion se quedó sin espacio y llegó
**truncado** (~4,8 KB de ~7 KB), así que el parser falló a mitad. **No** era el
timeout: esa corrida tardó ~85 s, por debajo de los 600 s.

**Arreglo:**

- `ollama.parametros.num_ctx` pasa a **16384** en la config local y en la
  plantilla `engines.example.json` (el 35b ya cabe: §5.8 lo midió en 12,2 GB con
  16384). Al reiniciar Ollama, `/api/ps` debe publicar `context_length: 16384`.
- `ScriptWorker` reconoce el aviso de truncación (`TRUNCATED_NOTE`) y lanza un
  error **accionable** («se agotó el contexto… sube `num_ctx` y pon
  OLLAMA_NUM_PARALLEL=1…») en vez del críptico `Expecting ',' delimiter`.

**Nota sobre el paralelismo:** `OLLAMA_NUM_PARALLEL=2` quedó del modo por lotes de
Documentos, que se **descartó** (`MAX_PARALLEL_PAGES = 1`). Con 2, Ollama reparte
el contexto entre slots y duplica el KV cache; para el guion conviene
**`OLLAMA_NUM_PARALLEL=1`** (`setx OLLAMA_NUM_PARALLEL 1` + reiniciar el servidor).

**Verificación:** `ruff check .` limpio, `num_ctx` = 16384 en local y plantilla, y
**52 tests en verde**.

### 10.3 El «cuadro de diálogo que aparece y desaparece» al escribir en el chat

**Síntoma:** al enviar un mensaje en el chat parpadeaba un diálogo pequeño (aparecía
y se cerraba solo) y la pestaña podía cambiar sola.

**Causa:** [`ui/main_window.py`](../../../Imperator/ui/main_window.py:336) muestra un
`QMessageBox.question` («¿Abrir la pestaña …?») cuando el router de intención detecta
otra tarea. Como el router se dispara **dentro del mismo evento** que el
`returnPressed` de la barra del chat, Qt le entregaba la tecla **Enter** al diálogo
recién abierto, que pulsaba el botón por defecto (**Sí**) y lo cerraba en el acto: de
ahí el parpadeo y el cambio de pestaña sin querer.

**Arreglo:** el diálogo se difiere al siguiente ciclo del bucle de eventos
(`QTimer.singleShot(0, …)` → `MainWindow._ask_open_tab`), de modo que la tecla ya no
está viva. La emisión de la señal `routed` **sigue siendo síncrona** (los tests de
`test_intent_router` la comprueban tal cual); solo se aplaza el mostrarlo.

**Verificación:** `ruff check .` limpio y **88 tests en verde** (`test_chat_widget`,
`test_intent_router`, `test_video_sequence`).

### 10.4 «Una pantalla negra que se esconde rápidamente»: consolas de subproceso

**Síntoma:** al enviar un mensaje en el chat parpadeaba una ventana negra, «como si
estuviera dando un comando a una terminal».

**Causa:** Imperator corre con `pythonw.exe` (sin consola) y **cada
`subprocess.run` abre una consola** en Windows. El `nvidia-smi` del planificador de
GPU ([`GpuProbe._run`](../../../Imperator/core/gpu_scheduler.py:132)) se sondea en
**cada reserva de GPU**, es decir, con solo enviar un mensaje en el chat. `ffmpeg`
(ensamblador, extracción de fotogramas y sondeos) hacía lo mismo.

**Arreglo:** nuevo
[`core/subprocess_utils.py`](../../../Imperator/core/subprocess_utils.py) con
`no_window_kwargs()` (`CREATE_NO_WINDOW` + `startupinfo` con `SW_HIDE`), aplicado en
**todos** los subprocesos: `gpu_scheduler`, `frame_extractor`, `video_assembler`
(probe y `_run`), `ui/audio_widget` y `ui/video_widget`.

**Verificación:** `ruff check .` limpio y **223 tests en verde**.

### 10.5 «Generar» en modo i2v pide fotos aunque lo que hay es un guion

**Síntoma:** tras generar el guion con sus escenas, pulsar **Generar** no hacía nada:
«Selecciona una o más imágenes para el modo Imagen a vídeo». El modo del widget se
persiste en `last_models.video` y está en `i2v`, así que **Generar** (clip único)
exige foto; las escenas se generan con **«Generar escenas»**.

**Arreglo:**
[`VideoWidget._generate`](../../../Imperator/ui/video_widget.py:1293) detecta que hay
un guion cargado y nombra el botón exacto: «Para las N escenas del guion usa “Generar
escenas”», en vez del mensaje genérico.

### 10.6 El widget no «sabía» que las escenas del guion ya estaban generadas

**Síntoma:** con las **6 escenas ya generadas** (el proyecto
`projects/el-secreto-del-trigo-dorado/` tenía `escena_00..05.mp4`, `foto_NN.png` y
`secuencia.mp4`), pulsar **Generar** seguía avisando del modo i2v y **Enviar al
Ensamblador** quedaba deshabilitado: parecía que el programa no sabía que ya estaba
hecho.

**Causa:** al generar el guion, `_on_script_ready` llamaba a `load_script(script)`
**sin proyecto**, así que `self._project`/`self._clips` quedaban vacíos: ni miniaturas
ni «Enviar al Ensamblador», aunque el proyecto existiera en disco.

**Arreglo:** `_on_script_ready` pasa `vproject.find_project(script.title)`, de modo que
un guion **con proyecto existente** se engancha (clips + `secuencia.mp4`) y el estado lo
dice («· N escena(s) ya generadas»). Además, `_generate` con todas las escenas hechas
indica el paso que falta («Usa “Enviar al Ensamblador”»). Test-guarda en
`tests/test_video_sequence.py` (`VideoScriptTableTest`).

**Verificación:** `ruff check .` limpio, **27 tests de vídeo** en verde, y la secuencia
real del proyecto de 6 escenas existe (`secuencia.mp4`, 13,4 MB, generada escena a
escena con «Fotos primero»).

### 10.7 Vídeo: «Nuevo» y «Borrar proyecto» (empezar otro vídeo desde cero)

**Síntoma:** al terminar un vídeo no había forma clara de empezar otro: la pestaña
seguía mostrando el anterior y no se veía un botón de borrado.

**Causa (dos cosas):**

1. El borrado **sí existía**, pero dentro de «Proyectos…» y solo sobre la fila
   seleccionada: sin selección, «Abrir»/«Borrar» no hacían **nada** y ni avisaban (el
   diálogo no seleccionaba ninguna fila al abrirse).
2. La pestaña Vídeo no tenía ningún botón para **empezar de cero**.

**Arreglo:**

- `ui/projects_dialog.py`: preselecciona la primera fila y, si no hay ninguna,
  «Abrir»/«Borrar» avisan («Selecciona un proyecto de la lista.»).
- `ui/video_widget.py`: botones **«Nuevo»** (limpia guion, clips y proyecto de la
  pestaña, con confirmación si hay clips; **no** borra nada del disco) y **«Borrar
  proyecto»** (borra del disco el proyecto cargado, con confirmación). Además,
  «Generar guion» limpia lo anterior antes de pedir el guion nuevo (un guion nuevo es
  un vídeo nuevo) y el proyecto que coincida por título se reengancha al llegar
  (§10.6).
- 4 tests nuevos en `tests/test_video_sequence.py` (nuevo, borrar, enganche del
  proyecto y preselección del diálogo).

**Verificación:** `ruff check .` limpio y **91 tests** en verde.

## 11. El guion no seguía la historia: el prompt del guionista

**Síntoma:** «el modelo no sigue la historia, se inventa cosas o no las relaciona
directamente».

**Causa:** el prompt del guionista pedía «una historia coherente», pero **no
obligaba** a que la trama saliera de la idea, ni a que cada escena fuese consecuencia
de la anterior, ni a que la narración y el `visual_prompt` mostrasen el mismo
instante. Sin esas restricciones, el modelo rellena con subtramas y elementos
inventados y las escenas quedan como viñetas sueltas (justo lo que se veía).

**Arreglo (solo prompt, `services/video_script.py`):**

- `SCRIPT_SYSTEM_PROMPT` pasa de 7 a **12 reglas** con cuatro que atacan el fallo:
  la **idea es el contrato** («no añadas tramas, personajes, objetos, lugares ni
  datos que no se deduzcan de ella … no lo inventes»), **cada escena es consecuencia
  directa de la anterior** («si se puede quitar sin que la historia cambie, no
  pertenece»), **continuidad** de lugar/momento del día/vestuario, y **narración =
  lo que se ve** (sin resúmenes ni moralejas).
- El mensaje de usuario añade un **repaso previo** con cinco comprobaciones (misma
  historia que la idea, cadena causal, un cambio real por escena, nada inventado,
  narración e imagen del mismo instante).
- El esquema JSON pide la relación causal también en `action` y aclara que
  `narration` describe lo que se ve.

**Verificación:** `ruff check .` limpio y **131 tests** verdes, con un test-guarda
nuevo (`tests/test_video_script.py::PromptTest`) que fija las cuatro reglas.

**Límite honesto:** esto **reduce** el fallo, no lo elimina. La coherencia sigue
siendo de prompt/estilo/semilla, y un modelo que razona puede desviarse. No hay forma
fiable de validar mecánicamente la «causalidad» de la historia sin otro modelo; si un
guion sale mal, el camino es editar la tabla (narración/prompt por escena) o
regenerar.

## 12. Editar una escena suelta (clip y foto)

**Petición:** poder cambiar cualquier clip por otro generado más tarde y cambiar la
foto de una escena para que el personaje sea otro.

**Arreglo:** menú contextual (clic derecho) en la tabla de escenas de la pestaña Vídeo:

- **Reemplazar clip (MP4)…** — copia el MP4 elegido (p. ej. de `output/`) a
  `escena_NN.mp4`, borra la miniatura cacheada y deja de ofrecer la `secuencia.mp4`
  guardada (ya no corresponde). «Generar escenas» la recompone **sin** rehacer las
  escenas ya hechas.
- **Reemplazar foto (PNG/JPG)…** — copia la imagen a `foto_NN.png`: es la vía para
  **cambiar el personaje** de esa escena (con «Fotos primero» esa foto es el primer
  fotograma de su clip).
- **Regenerar esta escena** — rehace **solo** ese clip con su foto actual:
  `SequenceWorker` gana el parámetro `only`, que limita el bucle a esa escena y **no
  arrastra** a las posteriores (antes, un hueco obligaba a rehacer la cola).
- **Quitar foto** — elimina `foto_NN.png`.

**Verificación:** `ruff check .` limpio y **77 tests** verdes, con 3 nuevos
(regenerar solo una escena, reemplazar clip y reemplazar foto).

## 13. Ampliar un guion ya escrito (escenas nuevas)

**Petición:** poder **aumentar** las escenas de un guion ya creado.

**Arreglo:** dos controles bajo la tabla de escenas de la pestaña Vídeo:

- **«Añadir escena»** — añade una escena **en blanco** al final (heading `ESC n`,
  prompts de relleno) para escribirla a mano en la tabla.
- **«Ampliar con IA»** (+ «Nuevas: N») — pide al modelo **solo las escenas nuevas**
  que continúen la historia. `services/video_script.py` gana
  `EXTENSION_SYSTEM_PROMPT` (continuar, no reescribir), `script_digest` (le da al
  modelo el guion ya escrito: estilo, reparto con su apariencia **literal** y las
  escenas), `extension_messages`, `parse_extension` y `extend_script` (añade al final
  y valida). El worker es `ScriptExtensionWorker`, bajo `gpu_slot('chat')`.

Al pulsar **«Generar escenas»**, la reanudación por prefijo contiguo genera **solo
las nuevas** (las ya hechas se conservan) y recompone la secuencia.

**Verificación:** `ruff check .` limpio y **83 tests** verdes, con 5 nuevos
(4 de `services/video_script` y 1 del worker de ampliación).

## 14. Validador ligero del guion (avisos deterministas, sin modelo)

**Decisión:** el chequeo es **determinista**, no una segunda llamada a un modelo. Es
rápido, gratis, testeable y no puede «alucinar»; lo que no puede es juzgar si la
historia tiene sentido (eso necesita un modelo o una persona).

**`services/video_script.py`:**

- `ScriptWarning` (frozen: `scene` 1-based o `None`, `code`, `message`) y
  `script_warnings(script)`.
- Comprueba: **narración vacía** o **repetida**; **lugar repetido** (mismo `heading`
  entre escenas consecutivas); **acción casi idéntica** (Jaccard de palabras ≥
  `ACTION_SIMILARITY` = 0,8); **salto brusco de momento del día** (mañana ↔ noche,
  leído del `heading`/`action`); **personaje que aparece por primera vez** en una
  escena > 1 y **personaje del reparto que no sale en ninguna escena**.
- Helpers puros: `_norm`, `_tokens`, `_day_moment`, `_DAY_MOMENTS`, `_DAY_JUMPS`.

**`ui/video_widget.py`:** etiqueta de avisos bajo la tabla (`⚠ N aviso(s): …`, en
ámbar) y marca de la fila afectada (fondo + tooltip con el detalle). **No bloquea
nada**: se recalcula al cargar el guion, al añadirlo/ampliarlo, al reordenar y al
editar una celda (`itemChanged`). `_rebuild_scenes` bloquea señales mientras rellena
las filas, y `_refresh_warnings` lleva un **guardia de reentrada** porque marcar una
celda emite `itemChanged` (sin él, `_refresh_warnings` se llamaba a sí misma hasta
**desbordar la pila**; se cazó en los tests).

**Verificación:** `ruff check .` limpio y **90 tests** verdes, con 7 nuevos (6 de
`services/video_script` y 1 del widget).

## 15. Streaming de /api/chat (progreso real y cancelación inmediata)

**Cierra el pendiente de §10.1 y §11.** La petición de chat era bloqueante: mientras
duraba (hasta el `timeout`) el botón **Cancelar** no hacía nada y no había señal de
avance.

**Arreglo:**

- `engines/base_engine.py`: `ChatEngine.chat_stream(messages, model, *, on_token,
  cancel_check)` con implementación por defecto que delega en `chat` (un motor que no
  lo soporte —o un doble de test— sigue funcionando igual). Nuevo helper
  `chat_engine_reply(engine, messages, *, model, on_token, cancel_check)`: el punto
  único que usan los workers y que elige streaming si el motor lo ofrece.
- `engines/ollama.py`: `OllamaEngine.chat_stream` pide `stream: true`, lee el NDJSON de
  `/api/chat` línea a línea, acumula `message.content`, llama a
  `on_token(delta, total_chars)` y consulta `cancel_check`: al cancelar se lanza
  `CANCEL_MESSAGE` y, al cerrarse la conexión, **Ollama detiene la generación**. El
  `thinking` no se emite (no es respuesta). Respeta `num_ctx`/`num_predict` y avisa de
  la respuesta cortada por contexto.
- `ui/chat_widget.py`: `ChatWorker` usa el helper y emite avance; el indicador
  «Pensando...» pasa a mostrar «Recibiendo… N caracteres».
- `ui/video_widget.py`: los workers de guion y de ampliación usan el helper; el estado
  muestra «Generando guion… recibiendo… N caracteres» y cancelar corta al instante.

**Verificación:** `ruff check .` limpio y **125 tests** verdes en los módulos tocados,
con 7 nuevos (4 de `chat_stream`, 2 del helper y 1 del worker de chat).

## 16. Prueba en caliente de la secuencia por escenas y fallo de transiciones (2026-10-04)

**Objetivo:** cerrar el pendiente de §9/§10 (probar en caliente una secuencia
larga con «Reanudar proyecto»). Se hizo **con evidencia, no re-ejecutando GPU**.

**Lo que dicen el disco y el log (comprobado):**

- `projects/` solo contiene `el-secreto-del-trigo-dorado/` y está **vacía**: no
  hay `escena_NN.mp4`, ni fotos, ni `secuencia.mp4`, ni `guion.json`/`proyecto.json`.
  El proyecto `el-secreto-del-trigo-de-basajaun` de §10 **no existe**.
- `logs/plataformaia.log` (03/10/2026) tiene la corrida **completa**: escena 1/6 a
  las 07:27:22 … escena 6/6 a las 07:50:40, cada clip terminado (`/history`
  `completed=True`) y `secuencia.mp4` concatenada a las 07:55:54. **Sin `ERROR`,
  sin `Traceback` y sin corte**: la liberación entre escenas de §9 aguantó en
  caliente, que es justo lo que el pendiente pedía confirmar.
- No hay ninguna línea de «reanudar»: la corrida fue limpia de la 1 a la 6, así
  que la ruta de reanudación **no se ejercitó en caliente** y ya no puede hacerse
  (no queda proyecto a medias). Sigue cubierta por los tests.

**Fallo real encontrado en esa corrida (y arreglado):** las **transiciones no se
aplicaban** y el ensamblado caía a corte seco:

```
WARNING [ui.video_widget] Transiciones no aplicadas (ffmpeg falló:
[Parsed_xfade_36] The inputs needs to be a constant frame rate;
current rate of 1/0 is invalid); corte seco.
```

**Causa (medida, no intuida):** en el grafo del ensamblador, `setpts=PTS-STARTPTS`
iba **después** de `fps=…`. Con ffmpeg 7.1 (`imageio-ffmpeg` 0.6 →
`ffmpeg-win-x86_64-v7.1.exe`) ese orden deja la tasa del enlace en **1/0** y
`xfade` la rechaza. Reproducido con los **6 clips reales** de `output/`
(720x1280 @16 fps) probando variantes del filtro: solo pasan `setpts` **antes**
de `fps`, o sin `setpts`.

**Arreglo:** [`core/video_assembler.py`](../../../Imperator/core/video_assembler.py:362)
construye la cadena como `setpts=PTS-STARTPTS,fps=…,format=yuv420p` (con
`scale`/`pad`/`setsar=1` delante). Test-guarda en
[`tests/test_video_assembler.py`](../../../Imperator/tests/test_video_assembler.py:171)
(`XfadeFilterTest.test_setpts_antes_de_fps`), que fija el orden.

**Verificación con los clips reales del 03/10:** el mismo ensamblado que fallaba
ahora produce `output/repro_transiciones.mp4` (6 escenas, **11,13 s, 30 fps,
5,5 MB**) **con las transiciones aplicadas** (fade, dissolve, cut, slide_left,
circle). Suite de vídeo **111 tests en verde** y `ruff check .` limpio.

**Nota para retomar:** como la carpeta del proyecto está vacía, una prueba en
caliente de punta a punta (Vídeo → Audio → Ensamblador → `video_final.mp4`)
exige regenerar las escenas (~4-5 min por clip en esta máquina). El código de ese
camino no ha cambiado en esta sesión.

## 17. Deuda A7 — duplicación entre `ollama` y `ollama_vision` (2026-11-18)

**Estado: pendiente. No se toca en esta ronda.** Anotado al cerrar la auditoría
del 18/11 y los commits de higiene.

`OllamaEngine` (`engines/ollama.py`) y `OllamaVisionEngine`
(`engines/ollama_vision.py`) mantienen tres métodos duplicados:

| Método | `engines/ollama.py` | `engines/ollama_vision.py` | Estado |
|---|---|---|---|
| `_apply_no_think` | :137 | :272 | copia idéntica |
| `is_loaded` | :327 | :283 | casi idéntico, ya divergió |
| `unload_model` | :340 | :294 | casi idéntico, ya divergió |

**No revertir `b6dbaab`.** Ese commit aporta features reales (cancelación,
timeout configurable y control de `think`). Lo que toca es extraer, no deshacer.

**La divergencia ya existe, no es hipotética:**

- El helper que explica una respuesta vacía tiene **dos nombres y dos
  comportamientos**: `_empty_reply_reason` (`engines/ollama.py:304`) y
  `_no_content_reason` (`engines/ollama_vision.py:228`). Cambia la firma
  (`data, message` frente a `message, razon`) y cambian las ramas: la de visión
  separa «`num_predict` gastado en razonar» de «ventana `num_ctx` agotada»
  (`ollama_vision.py:238-257`), mientras que la del chat mete los dos casos en un
  mismo mensaje (`ollama.py:312-316`). Una corrección en una no llega a la otra.
- `is_loaded` del chat toma `self._lock` (`ollama.py:334`) y usa
  `self._short_timeout`; la de visión **no toma lock** y usa
  `self._SHORT_TIMEOUT` (`ollama_vision.py:289-292`). El mismo patrón de
  protección no se aplicó dos veces.
- `unload_model` repite esa asimetría de lock y de timeout, y cambia el texto del
  log (`ollama.py:340-367` frente a `ollama_vision.py:294-318`).

**Solución propuesta:** extraer a un mixin `_OllamaCommon` (`_apply_no_think`,
`is_loaded`, `unload_model` y el helper de respuesta vacía unificado) o, si se
prefiere no tocar la jerarquía, mover lo común a `engines/ollama_api.py`, que ya
existe y ya centraliza la consulta de modelos cargados.

**Impacto:** cada corrección hay que hacerla dos veces, y **dos de las tres
copias ya han divergido**. La ronda debe traer tests propios del mixin (una sola
`is_loaded`, un solo mensaje de respuesta vacía) y cerrar con
`IMPERATOR_GUI_SMOKE=1 python -m unittest discover -s tests`.

**Por qué se aplaza:** es una ronda propia (mixin + tests). Mezclarla con C4
(`settings.json` inválido tumba `MainWindow`) y C5 (E/S en el hilo de UI) diluye
el foco de las dos que hoy se notan en uso real.
