# Vídeo por escenas: del chat al montaje final

Documento de referencia del **pipeline de guion y generación de vídeo por
escenas** de Imperator. Describe qué se construyó, cómo fluyen los datos, las
decisiones de diseño y los límites reales.

## 1. Para qué sirve

Convertir una idea escrita en el chat en una **secuencia de clips coherente**,
con una historia que se sostiene y **los mismos personajes** en todas las
escenas, sin que el usuario reescriba el guion escena por escena ni aprenda a
usar el generador de vídeo.

Frase de ejemplo:

> «Haz un guion de 4 escenas sobre un viajero en un bosque.»

debe generar el guion, ponerlo en una tabla editable y dejar la generación a un
clic, sin pasos intermedios manuales.

## 2. El flujo completo

```
📬 CHAT            "guion de 4 escenas sobre..."
   │               intent_router (regla ACTION_PLAN_SCRIPT)
   ▼
🎯 DISPATCH        Dispatch{task_id:"video", action:"plan_script",
   │                        payload:{idea, scenes}}
   ▼
📋 VÍDEO           MainWindow abre la pestaña y llama a start_script()
   │               ScriptWorker → Ollama → JSON → Script validado
   │               tabla de escenas EDITABLE (narración, prompt, duración)
   ▼
🎬 GENERACIÓN      SequenceWorker: escena 1 → escena 2 → ...
   │               semilla compartida + apariencia fija + encadenado i2v
   │               cada escena se guarda al terminar y se libera la memoria
   │               (VRAM del motor + RAM ociosa) antes de la siguiente
   ▼
🎞️ MONTAJE         concat (corte seco) o `xfade` (con transiciones)
   │               proyecto persistente: projects/<slug>/
   ▼
✅ ENTREGA         "Enviar al Ensamblador" precarga la secuencia;
                   Audio (Piper/Kokoro) y Ensamblador cierran el vídeo.
```

El paso de narración con **Piper/Kokoro** y el montaje final con subtítulos y
música siguen siendo las pestañas **Audio** y **Ensamblador** de siempre: este
pipeline entrega las **escenas** y la secuencia, no sustituye al montaje manual.

## 3. Piezas nuevas (todas con tests)

| Archivo | Responsabilidad |
| --- | --- |
| [`services/video_script.py`](../Imperator/services/video_script.py) | Modelo puro del guion (`Script`/`Scene`/`Character`), validación, JSON y **prompts** (guionista, esquema y coherencia de personajes). |
| [`services/video_project.py`](../Imperator/services/video_project.py) | Proyecto persistente: carpeta autocontenida con `guion.json`, un MP4 por escena, **una foto por escena** (`foto_NN.png`), narración, SRT y montaje. Vive en `projects/`, **fuera** de la retención de `output/`. |
| `services/video_params.py` (ampliado) | Continuidad: `new_base_seed()`, `scene_seed()` y `scene_generate_params()` (prompt coherente + semilla + encadenado i2v + **tamaño del widget**), y `scene_image_params()` para la **foto** de cada escena (mismo prompt y semilla que su clip). |
| `services/intent_router.py` (ampliado) | Acción `ACTION_PLAN_SCRIPT` y reglas de guion («guion de un video», «storyboard», «video con escenas»…). |
| `services/task_orchestrator.py` (ampliado) | `Dispatch.payload` con `{idea, scenes}` y `_script_payload()`; `parse_scene_count()` resuelve el «de 4 escenas». |
| `services/chat_params.py` (ampliado) | `with_system_prompt()`: inyecta el rol sin duplicar el mensaje de sistema. |
| `core/video_assembler.py` (ampliado) | `probe()` devuelve `fps`; `xfade_filter_complex()` (grafo puro y testeable) y `concat_clips_with_transitions()`. |
| `ui/video_widget.py` (ampliado) | `ScriptWorker`, `SequenceWorker` (guardado por escena, **reanudación**, **liberación de memoria** y **«Fotos primero»**), tabla de escenas editable, botón «Generar guion», opciones de transiciones, «Reanudar proyecto» y **selector de motor/modelo de imagen**, y señal `clips_ready`. |
| [`core/memory.py`](../Imperator/core/memory.py) | Liberación de memoria entre escenas: `gc.collect()` + recorte del *working set* de Windows (devuelve al SO la RAM ociosa). Best-effort y con la parte dependiente del SO inyectable. |
| `ui/projects_dialog.py` | Cola de proyectos: lista con miniatura, **abrir** (carga el guion en la tabla) y **borrar** (confirmado). |
| `ui/assembler_widget.py` (ampliado) | `set_video()` para precargar la secuencia enviada desde la pestaña Vídeo. |
| `ui/main_window.py` (ampliado) | Empalmes desacoplados: `start_script` desde el chat y `clips_ready` → Ensamblador. |

## 4. Decisiones de diseño

### 4.1 Coherencia: tres capas

1. **Estilo único** (`Script.style`): una "biblia visual" que se antepone a
   **todos** los planos.
2. **Personajes con apariencia bloqueada** (`Character.appearance`): la
   descripción física se repite **literal** en cada escena donde aparece. Es lo
   que mantiene al personaje igual entre clips (`melena negra, gabardina ocre`
   da el mismo resultado; `una mujer alta` no).
3. **Semilla compartida** (`scene_seed`): todos los planos derivan de una misma
   semilla base; cada escena tiene la suya (distinta, reproducible).

### 4.2 Encadenado i2v (continuidad real)

Con el encadenado activo, `SequenceWorker` extrae el **último fotograma** de
cada clip (`core.frame_extractor`) y lo usa como imagen de partida de la
siguiente escena (modo `i2v`). Es la continuidad visual de verdad; el precio es
la **deriva acumulativa** en cadenas largas (por eso conviene re-anclar cada N
escenas).

### 4.3 El guion es JSON, no texto libre

El motor recibe el rol de guionista (`SCRIPT_SYSTEM_PROMPT`) y un **esquema
JSON**. `parse_script()` es tolerante (vallas de código, texto alrededor) y
`validate()` rechaza guiones imposibles: duración fuera del rango de Wan (2-5 s),
escenas sin descripción o personajes referenciados que no están en el reparto.

### 4.4 «Editar» es editar el guion, no el píxel

La frontera deliberada: la automatización llega hasta las **escenas**. La tabla
es editable (narración, prompt, duración) y **volver a generar una escena no
toca las demás**, pero la edición fina del vídeo renderizado (timeline, recorte
frame-accurate) queda fuera: eso es un NLE y no aporta proporcionalmente. El
montaje final se hace en **Audio** y **Ensamblador**.

### 4.5 Transiciones opt-in

`concat_clips` une sin recodificar (corte seco, rápido). Con la casilla
**Transiciones** activada se usa `xfade`, que **normaliza y recodifica** cada
clip (tamaño/fps/pix_fmt) y encadena. Un «corte» se implementa como cruce de un
fotograma. Si el `xfade` falla, se cae a `concat_clips` avisando en el log: es
preferible entregar la secuencia unida que no entregar nada.

### 4.6 El proyecto no vive en `output/`

`output/` tiene **retención** (`services/output_cleanup.py`): borra lo viejo. Los
intermedios de un trabajo (un MP4 por escena, narración, SRT) no pueden estar
ahí o desaparecen a mitad de edición. Por eso cada trabajo es un **proyecto** en
`projects/<slug>/`, que la retención **no toca**.

### 4.7 Nada de Qt ni de motores en la lógica

Todo lo decidible sin GUI (modelo de guion, continuidad, proyecto, grafo de
`xfade`) es **puro y testeable con dobles**: los tests no necesitan GPU, red ni
ffmpeg. Qt solo orquesta (`QThread`) y los motores solo infieren.

### 4.8 Reanudar en vez de reempezar (y liberar entre escenas)

Una secuencia larga puede morir a mitad (memoria, un cuelgue de ComfyUI, un corte
de luz). Antes, volver a pulsar **Generar escenas** rehacía todo desde la escena
1 y volvía a fallar en el mismo punto. Ahora:

- **Cada escena se guarda al terminar** en `projects/<slug>/escena_NN.mp4`. El
  guion (`guion.json`) y la **semilla común** (`base_seed`, en `proyecto.json`)
  se persisten con el proyecto.
- **Reanudación** (casilla «Reanudar proyecto», encendida por defecto): al volver
  a lanzar se salta el **prefijo contiguo** de escenas ya hechas y sigue por la
  primera que falta, reconstruyendo el encadenado i2v desde el último fotograma
  guardado y **reutilizando la misma semilla** (misma identidad visual). Un hueco
  invalida lo posterior —se regenera desde la primera que falta— porque el
  encadenado dejaría de ser continuo.
- **Liberación entre escenas** ([`core/memory.py`](../Imperator/core/memory.py)):
  tras guardar cada escena se pide al motor que suelte su VRAM
  (`unload_model`), se borra el fotograma de encadenado ya consumido y se
  devuelve al sistema operativo la RAM ociosa del proceso (`gc.collect()` +
  `EmptyWorkingSet` en Windows). Es lo que evita que el proceso crezca escena a
  escena hasta que Windows lo mata.
- **El tamaño es el del widget**: `scene_generate_params(width, height)` usa el
  ancho/alto elegidos en la pestaña (antes la generación por escenas ignoraba
  esos controles y usaba solo el preset por defecto del formato).

### 4.9 Fotos primero: cada escena con su foto y su prompt

El guion ya da un **prompt visual por escena** (columna «Prompt visual»). La
casilla **«Fotos primero (imagen → vídeo)»** da el paso que faltaba: antes de
cada clip se genera su **foto** con el motor de imagen elegido (ComfyUI local o
BFL de pago, en el selector de al lado) usando **ese mismo prompt** y la
**semilla de la escena** (`scene_image_params`), y esa foto entra como primer
fotograma del clip **i2v**.

- **Una foto = un prompt = un clip.** No hay que escribir nada dos veces: lo que
  se pide a la imagen es exactamente lo que arranca el vídeo.
- **Se guarda en el proyecto** (`foto_NN.png`) y **se reutiliza** si ya existe:
  reanudar no vuelve a gastar GPU en las fotos hechas, y el usuario puede
  sustituir una a mano antes de generar su clip.
- **Sin cadena**: con fotos, cada escena arranca de la suya (el fotograma del
  clip anterior no se usa), así que **cada escena es independiente**: un clip
  malo se regenera solo (basta borrar su `escena_MP4`/`foto_PNG`).
- **Progreso honesto**: con fotos, cada escena son dos pasos (foto y clip) y la
  barra los refleja, pero el texto siempre habla de la escena.
- Coste: una generación de imagen por escena (~10-25 s con `flux_dev`). Si no
  hay motor de imagen disponible, se avisa y se genera sin fotos (como antes).

## 5. Flujo desde el chat (payload del `Dispatch`)

El chat **no** genera el guion ni abre pestañas: solo emite `routed`.
`services.task_orchestrator` decide y, para un guion, adjunta:

```python
Dispatch(task_id="video", action="plan_script",
         payload={"idea": "<texto del usuario>", "scenes": 4})
```

`parse_scene_count()` extrae el «de 4 escenas» (es/eu/en). `MainWindow._on_chat_routed`
entonces abre la pestaña Vídeo y llama a `VideoWidget.start_script(idea, scenes)`,
que **reutiliza el mismo camino** que el botón «Generar guion». Sin payload, el
comportamiento es el de siempre: ofrecer abrir la pestaña con un diálogo.

## 6. Uso

1. **Chat:** «Haz un guion de 4 escenas sobre un viajero en un bosque.»
   (o, en la pestaña Vídeo, escribe la idea y pulsa **Generar guion**).
2. La **tabla** muestra las escenas; edita narración, prompt o duración.
3. Opcional: marca **Fotos primero (imagen → vídeo)** y elige el motor/modelo de
   imagen (ComfyUI `flux_dev`/`flux_schnell`… o BFL). Cada escena se generará
   como foto y esa foto arrancará su clip.
4. Pulsa **Generar escenas** (opción **Transiciones** a tu gusto).
5. Revisa el progreso (escena a escena) y, al terminar, **Enviar al Ensamblador**.
6. En **Audio** genera la narración; en **Ensamblador** monta clips + voz +
   subtítulos + música.

Los artefactos quedan en `projects/<slug>/`; el vídeo final del Ensamblador, en
`output/`.

## 7. Límites conocidos

- **Clips cortos.** Wan genera 2-5 s por escena: un vídeo de 60 s son muchas
  escenas y bastante tiempo de GPU.
- **La coherencia es de prompt/estilo/semilla**, no identidad facial real. El
  encadenado i2v ayuda pero **arrastra deriva**; conviene re-anclar.
- **Regenerar una escena intermedia** de una cadena i2v descoloca las
  posteriores (hay que re-generar la cola o re-anclar).
- **Reanudar no repara una escena mala.** Las escenas ya guardadas no se
  regeneran: para revisar una concreta, desmarca «Reanudar proyecto» (rehace la
  secuencia) o borra su clip del proyecto.
- **Ollama es opcional**: sin él, la pestaña Vídeo sigue generando clip a clip
  (solo no puede generar el guion desde el chat/botón).

## 8. Mejoras de uso

- **Miniaturas por escena.** Al generar (o al abrir un proyecto) cada fila muestra
  el primer fotograma de su clip (`FrameExtractor`), cacheado como
  `miniatura_NN.png` dentro del proyecto.
- **Reordenar arrastrando.** La tabla usa `InternalMove`; al soltar, el orden se
  aplica con `Script.move_scene()` y se avisa de que la coherencia i2v puede
  resentirse.
- **Modelo por guion.** Un combo elige el modelo de Ollama que escribe el guion
  (`ScriptWorker.model`); «(modelo por defecto)» delega en el motor.
- **Tipos de transición.** Además de corte/fundido/disolvencia: deslizar, barrido,
  círculo y zoom (`TRANSITIONS` ↔ `XFADE_NAMES`, con un test que garantiza la
  cobertura). El combo los aplica a todos los cruces o respeta «(por escena)».
- **Config por proyecto.** `Script.config` (fps, transiciones y tipo) se guarda
  con el guion y se restaura al reabrirlo.
- **Cola de proyectos.** «Proyectos…» abre [`ui/projects_dialog.py`](../Imperator/ui/projects_dialog.py):
  lista con miniatura, **Abrir** (carga guion y clips en la tabla) y **Borrar**
  (confirmado). El borrado es explícito porque `projects/` está fuera de la
  retención de `output/`.
- **Editar una escena suelta (clic derecho en su fila).** **Reemplazar clip** (un MP4
  que ya tengas, p. ej. de `output/`), **Reemplazar foto** (cambia la imagen de
  arranque: es la vía para cambiar el personaje de esa escena), **Regenerar esta
  escena** (rehace **solo** ese clip, con su foto actual, sin tocar las demás) y
  **Quitar foto**. Al reemplazar un clip, la `secuencia.mp4` guardada deja de
  ofrecerse hasta que «Generar escenas» la recomponga (no rehace las escenas hechas).
- **Nuevo / Borrar proyecto.** En la pestaña Vídeo: **«Nuevo»** limpia guion, clips y
  proyecto de la pestaña para empezar otro vídeo (no borra nada del disco) y **«Borrar
  proyecto»** elimina del disco el proyecto cargado (confirmado).
- **Ampliar el guion.** Bajo la tabla: **«Añadir escena»** (en blanco, para escribirla
  a mano) y **«Ampliar con IA»** («Nuevas: N»), que pide al modelo **solo las escenas
  nuevas** que continúen la historia, con el mismo reparto y estilo. «Generar escenas»
  genera únicamente las que falten.
- **Avisos del validador ligero.** Bajo la tabla aparece un aviso en ámbar cuando el
  guion tiene problemas **medibles sin modelo**: narración vacía o repetida, lugar o
  acción repetidos entre escenas seguidas, salto brusco de momento del día (mañana ↔
  noche) o personajes que aparecen de la nada / no salen nunca. La fila afectada se
  marca con fondo y tooltip. **No bloquea nada**: solo orienta, y se recalcula al
  editar la tabla.

## 9. Pruebas

- `tests/test_video_script.py` — modelo, validación, JSON, prompts,
  `parse_script`, `transition_sequence`, `parse_scene_count`.
- `tests/test_video_pipeline.py` — proyecto persistente, `base_seed`/`find_project`,
  `resume_index` (prefijo contiguo) y continuidad.
- `tests/test_video_sequence.py` — `SequenceWorker` (semilla/i2v/transiciones,
  **reanudación** y **liberación de memoria entre escenas**), `ScriptWorker`
  (parseo del guion) y la tabla del widget.
- `tests/test_memory.py` — contrato de `core.memory.release_memory` (recolecta,
  trim inyectado una vez y nunca lanza).
- `tests/test_video_assembler.py` — grafo `xfade` (puro).
- `tests/test_intent_router.py` — enrutado de guion y `payload` del `Dispatch`.

Todos con dobles: sin GPU, sin red y sin ffmpeg.
