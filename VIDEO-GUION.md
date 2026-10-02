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
| [`services/video_project.py`](../Imperator/services/video_project.py) | Proyecto persistente: carpeta autocontenida con `guion.json`, un MP4 por escena, narración, SRT y montaje. Vive en `projects/`, **fuera** de la retención de `output/`. |
| `services/video_params.py` (ampliado) | Continuidad: `new_base_seed()`, `scene_seed()` y `scene_generate_params()` (prompt coherente + semilla + encadenado i2v). |
| `services/intent_router.py` (ampliado) | Acción `ACTION_PLAN_SCRIPT` y reglas de guion («guion de un video», «storyboard», «video con escenas»…). |
| `services/task_orchestrator.py` (ampliado) | `Dispatch.payload` con `{idea, scenes}` y `_script_payload()`; `parse_scene_count()` resuelve el «de 4 escenas». |
| `services/chat_params.py` (ampliado) | `with_system_prompt()`: inyecta el rol sin duplicar el mensaje de sistema. |
| `core/video_assembler.py` (ampliado) | `probe()` devuelve `fps`; `xfade_filter_complex()` (grafo puro y testeable) y `concat_clips_with_transitions()`. |
| `ui/video_widget.py` (ampliado) | `ScriptWorker`, `SequenceWorker`, tabla de escenas editable, botón «Generar guion», opción de transiciones y señal `clips_ready`. |
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
3. Pulsa **Generar escenas** (opción **Transiciones** a tu gusto).
4. Revisa el progreso (escena a escena) y, al terminar, **Enviar al Ensamblador**.
5. En **Audio** genera la narración; en **Ensamblador** monta clips + voz +
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

## 9. Pruebas

- `tests/test_video_script.py` — modelo, validación, JSON, prompts,
  `parse_script`, `transition_sequence`, `parse_scene_count`.
- `tests/test_video_pipeline.py` — proyecto persistente y continuidad.
- `tests/test_video_sequence.py` — `SequenceWorker` (semilla/i2v/transiciones),
  `ScriptWorker` (parseo del guion) y la tabla del widget.
- `tests/test_video_assembler.py` — grafo `xfade` (puro).
- `tests/test_intent_router.py` — enrutado de guion y `payload` del `Dispatch`.

Todos con dobles: sin GPU, sin red y sin ffmpeg.
