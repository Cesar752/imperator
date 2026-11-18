# Imperator (PlataformaIA)

Plataforma modular de IA con backend Python y GUI PyQt6: unifica motores locales (Ollama, llama.cpp, LM Studio, ...), modelos (LLM, imagen, vídeo, audio, multimodal), APIs externas (OpenAI, DeepSeek, Kling, ...) y tareas tipo plugin (chat, OCR, imagen, vídeo, TTS/STT, documentos).

## Vídeo por escenas

Pide un guion desde el chat y la pestaña **Vídeo** lo convierte en una secuencia
coherente: «Haz un guion de 4 escenas sobre un viajero en un bosque». El chat
detecta la intención, genera el guion (Ollama), lo muestra como escenas
**editables** y lo genera con los **mismos personajes** (apariencia fija,
semilla compartida y encadenado *i2v*). Los artefactos quedan en
`projects/<slug>/` (fuera de la retención de `output/`) y **Enviar al
Ensamblador** cierra el montaje.

Guía completa: [VIDEO-GUION.md](VIDEO-GUION.md). Arquitectura: [ARCHITECTURE.md](ARCHITECTURE.md).

## Integrar programas externos (CLI)

Además de los motores nativos, Imperator puede invocar **programas de línea de
comandos** como motores. Se configuran en `engines.json` con el ejecutable
(`ruta` o `parametros.executable`) y una **plantilla de argumentos**; los huecos
disponibles son `{prompt}`, `{input}`, `{output}`, `{output_dir}` y `{model}`.
La app rellena `{output_dir}`/`{output}` (nombre único por corrida) y recoge el
artefacto que produzca la herramienta.

```json
{
  "id": "bkingfilm",
  "nombre": "BkingFilm (CLI)",
  "ruta": "C:/ruta/a/bkingfilm.exe",
  "enabled": true,
  "parametros": {
    "args": ["--input", "{input}", "--prompt", "{prompt}", "--output", "{output}"],
    "output_ext": ".mp4",
    "timeout": 600
  }
}
```

`bkingfilm` y `lapian_notes` vienen ya registrados como tipo de motor (en
`core/engine_manager.py`) y **deshabilitados**: si el ejecutable no está
instalado —o no se encuentra en el PATH— el motor aparece como **no
disponible**, sin romper el arranque. `lapian_notes` se integra igual (analiza
un MP4 y produce el paquete de análisis). La ejecución oculta la consola
(`core/subprocess_utils.py`), aplica `timeout` y traduce los fallos a errores de
motor. La lógica vive en [`core/cli_runner.py`](core/cli_runner.py) y
[`engines/cli_tool.py`](engines/cli_tool.py).

Todo se ejecuta **en local**: no se sube nada a ningún servidor. Si prefieres no
tocar el `PATH`, coloca el ejecutable en `tools/` o `bin/` del propio proyecto:
se resuelve solo (también admite `.exe`/`.bat` de Windows).

## Cuaderno (Lapian Notes)

Análisis cinematográfico **local**. En la pestaña **Cuaderno** eliges un vídeo
(MP4, MKV, MOV, **AVI**, WebM, M4V o los **MTS/M2TS** de cámara: lo que decodifica
ffmpeg) y:

- **Generar paquete (ZIP)**: extrae un fotograma por segundo (ffmpeg) y monta un
  ZIP con los fotogramas, las instrucciones y el esquema JSON para una IA.
- **Analizar en local**: describe los fotogramas con Ollama Vision y compone el
  JSON del cuaderno con Ollama (chat), sin pasar por ChatGPT.
- **Importar análisis (JSON)**: carga el resultado de cualquier IA.
- **Exportar Markdown**: cuaderno con líneas narrativas, estructura, curva de
  emoción y segmentos/escenas.

La película no sale de tu equipo. La lógica pura está en
[`services/notes_params.py`](services/notes_params.py) y la pestaña en
[`ui/notes_widget.py`](ui/notes_widget.py).

## Crónica (vídeo comentado por IA)

Pestaña **Crónica**: eliges un partido y la app lo **mira, lo cuenta, lo locuta y
lo monta**, todo en local. Cinco pasos, encadenados en un solo botón:

1. **Mira**: describe fotogramas con Ollama Vision (y, si quieres, transcribe el
   audio con Whisper).
2. **Cuenta**: el motor de chat escribe frases cortas **con el segundo en el que se
   dicen** (JSON con `t_s`; si contesta en texto, se acepta «minuto:segundo frase»).
3. **Locuta**: cada frase se sintetiza con Piper/Kokoro/XTTS y se coloca en su
   segundo con `adelay`; todo se mezcla en una sola pista con `amix`.
   - **El sonido de la grabación se conserva** bajo la voz (casilla «Conservar el
     audio original», activada por defecto): se sigue oyendo el público, el silbato
     y el balón. Desmárcala si quieres oír solo la narración.
4. **Música** (opcional y **apagada por defecto**): MusicGen genera la banda sonora y
   el Ensamblador la mezcla con *ducking*. Si no la quieres, no hay nada que hacer:
   sin casilla marcada no se añade ninguna pista de música.
   - Al montar, el relleno de silencio se acota a la duración del vídeo
     (`apad=whole_dur`). Sin acotarlo, ffmpeg 7.1 se queda generando silencio para
     siempre al copiar el vídeo y el ensamblado **se cuelga** (medido: más de 10
     minutos; acotado, 0,1 s).
5. **Monta**: subtítulos quemados + voz + música sobre el vídeo, y deja una
   `*_cronica.md` con el texto y sus tiempos.

Pensado para balonmano: de una grabación se pasa a un vídeo comentado sin editar
nada a mano (y sin subir el partido a ningún sitio). Lógica pura en
[`services/chronicle_params.py`](services/chronicle_params.py), pestaña en
[`ui/chronicle_widget.py`](ui/chronicle_widget.py).

Con la **alineación** en «Contexto»/«Datos del partido» (una línea
`Alineación <equipo>: <dorsal> <nombre>, …`), el programa llama a cada jugadora
por su **nombre de pila**: traduce «dorsal 7» / «la del rojo» al nombre cruzando
el dorsal con la alineación (vale el nombre completo del acta). Sin alineación no
inventa nombres. La lógica está en
[`services/roster_params.py`](services/roster_params.py).

Para que **Comprensión, Cuaderno y Crónica compartan los mismos minutos**, cada
frase de la crónica se ancla al instante `[MM:SS]` anotado en el informe (el modelo
debe devolver su `ref`, y la app lo coloca en esa marca): así la narración cae en
el momento real del hecho y no se inventa ni reparte tiempos. El anclaje está en
[`services/chronicle_params.py`](services/chronicle_params.py) (`parse_timestamp`,
`anchor_lines`).

La **disciplina** (balonmano/genérico) es **una sola preferencia** compartida con
**Comprensión** y el **Cuaderno** ([`ui/discipline.py`](ui/discipline.py)): cambiarla
en cualquier pestaña la cambia en las demás, así el flujo
**Comprensión → Cuaderno → Crónica** habla siempre del mismo deporte. Las reglas viven
en [`services/sport_params.py`](services/sport_params.py).

### Elegir la IA de la crónica (local o de pago)

La crónica la firman **dos modelos**, elegibles en la propia pestaña. Tanto el motor
como el **modelo** se pueden elegir **en local o de pago**: la lista de modelos trae
los configurados y los que ofrezca el motor —los **instalados en Ollama/LM Studio** y
los de la **API** (OpenAI/DeepSeek…)—, así que se cambia de uno a otro sin editar
`engines.json`.

- **Visión**: quién mira los fotogramas. Por defecto, **Ollama Vision** (local y
  gratis). Para una descripción más fina se puede elegir **OpenAI Vision**
  ([`engines/openai_vision.py`](engines/openai_vision.py), cualquier API compatible
  con OpenAI: GPT-4o, Gemini por su endpoint compatible, xAI…), que ve mejor pero
  **cobra por fotograma** (la pestaña avisa del gasto). Requiere `OPENAI_API_KEY` en
  `.env` y el motor habilitado en `config/engines.json`. Sólo aparecen motores que
  **saben mirar** (`analyze_image`): el modelo tiene que ser **multimodal**.
- **Chat**: quién escribe el guion. Por defecto, el motor elegido en la pestaña
  **Chat**; aquí se puede usar otro (p. ej. OpenAI o DeepSeek) sólo para la crónica.
  **DeepSeek no tiene visión** (`deepseek-chat`/`deepseek-reasoner` son de texto):
  sirve para **redactar**, no para mirar los fotogramas; por eso nunca aparece en el
  selector «Visión». Para visión local de DeepSeek está DeepSeek-VL vía Ollama.

La forma **más barata y precisa** de generar la crónica es reutilizar lo que ya
analizaron **Comprensión** y **Cuaderno**: con «Usar percepción de Comprensión…»
(`comprension.json`) o «Usar informe del Cuaderno…» (`informe.json`) marcado, la
crónica se escribe **sin volver a mirar el vídeo** (ni visión ni Whisper), así que no
gasta GPU ni tokens de visión. La pestaña lo indica en pantalla, y el flujo
recomendado es **Comprensión → Cuaderno → Crónica**.

### Decirle a la IA qué contar (instrucciones)

Comprensión, Cuaderno y Crónica tienen una caja **«Instrucciones para la IA»**: lo
que quieres que cuente y en qué orden. Se inyecta en el prompt del **texto** que
compone esa pestaña (el resumen, el cuaderno o el guion de la crónica) y se recuerda
entre sesiones. **No** se pasa a la visión de cada fotograma: ahí el modelo intentaba
«presentar la alineación» en todos los fotogramas y alucinaba listas enormes. La
visión por fotograma sólo recibe el **mapa color → equipo (LOCAL/VISITANTE)** para no
confundir ataque y defensa.
El botón **«Poner ejemplo (balonmano)»** rellena un punto de partida recomendado:
presentar la **alineación** al principio (mientras calientan) y luego centrarse en
**goles, paradas del portero y lanzamientos de 7 metros**.

Además, la Crónica ya prioriza esas jugadas decisivas (goles, paradas y 7 metros) y
su visión pide el **estado** de cada fotograma (PREVIO/CALENTAMIENTO/JUGANDO/SAQUE/
GOL/PARADA/7 METROS/TIEMPO MUERTO/DESCANSO/FINAL) y **qué equipo tiene el balón**
(ese ataca; el otro defiende), para que las paradas no se pierdan por describir solo
«jugadores y balón».

Correcciones de precisión de la narración:

- **Cada frase se queda en su momento**: sólo se corrige la deriva si hay una
  anotación a ±3 s (`place_lines`); antes `anchor_lines` movía cada frase al instante
  libre más cercano y una frase del calentamiento podía acabar en el minuto 3:40.
- **Apertura sin solapes**: las frases de sede/fecha/alineaciones se encadenan por la
  **duración real del audio** ya sintetizado (no por una estimación), así no se pisa
  la alineación de un equipo con la del otro.
- **Marcador con nombres**: de «Equipos: … (local) vs … (visitante)» se deduce quién
  es local y visitante (izquierda/derecha del marcador) y el **marcador real** del OCR
  se etiqueta con los nombres; el guion tiene prohibido decir que va por delante un
  equipo que no lo esté en esa serie.
- **Goles y paradas**: el guion dedica **una frase a cada** anotación con GOL, PARADA
  o 7 METROS del informe, y el muestreo por defecto de la Crónica baja a **un
  fotograma cada 10 s** (antes 30 s, con el que se perdían jugadas enteras).

La crónica **abre siempre** con una **apertura determinista** compuesta con reglas a
partir de «Datos del partido»: la **sede** (polideportivo/lugar), la **fecha** y la
**alineación de los dos equipos**. No depende del modelo ni de que exista una
anotación `[MM:SS]` en ese momento (`opening_lines` + `prepend_opening` en
[`services/chronicle_params.py`](services/chronicle_params.py)); si alguna frase del
modelo empezara durante la apertura, se descarta para no solapar la locución.

## Clonar voz (XTTS-v2)

La pestaña **Audio** incluye el motor **«Clonar voz (XTTS-v2)»**: con unos
segundos de audio de referencia en `voices/clones/`, sintetiza texto con esa voz
(clonado *zero-shot*). Es **opcional**: sin sus dependencias aparece como *no
disponible* y no afecta al arranque.

Coqui va en un **venv aparte** para no arrastrar torch/transformers al arranque
de la app. El fork mantenido (`coqui-tts`) sí soporta Python 3.12:

```powershell
py -3.12 -m venv venv-tts
venv-tts\Scripts\python.exe -m pip install "coqui-tts[codec]" "transformers<5" torchaudio
setx COQUI_TOS_AGREED 1   # acepta la licencia de pesos (una sola vez)
```

(`[codec]` instala `torchcodec`, que torch ≥2.9 exige para el audio IO, y
`transformers<5` porque Coqui usa `isin_mps_friendly`, eliminado en la 5.) El
modelo XTTS-v2 (~1,8 GB) se descarga la primera vez que se sintetiza.

Añade tus muestras a `voices/clones/` (`.wav`/`.mp3`/…), activa el motor en
*Motores → Gestionar motores…* (`voice_clone`, con `language` y `device`) y elige
la muestra en el desplegable **Voz**.

El motor **no importa Coqui en Imperator**: por defecto usa `backend: "subprocess"`
y lanza [`tools/xtts_worker.py`](tools/xtts_worker.py) con el intérprete del venv
(`parametros.python`, por defecto `venv-tts/Scripts/python.exe`). Así el clonado
funciona aunque la app use Python 3.12. Motor en
[`engines/voice_clone.py`](engines/voice_clone.py).

## APIs externas (OpenAI / DeepSeek)

El **chat** puede usar una API externa compatible con OpenAI. Actívala en
*Motores → Gestionar motores…* (`openai` o `deepseek`) y pon la clave en `.env`:

```dotenv
OPENAI_API_KEY=...
DEEPSEEK_API_KEY=...
```

El motor ([`engines/openai_compatible.py`](engines/openai_compatible.py)) habla
`POST {base_url}/chat/completions` con `Authorization: Bearer <clave>`; las
claves se leen **solo del entorno** (nunca del JSON). En la pestaña **Chat**
aparece un selector **Motor** (Ollama / OpenAI / DeepSeek) para cambiar de uno a
otro sin reiniciar. `is_available()` sólo comprueba que haya clave (no hace red).

La elección de motor es **global**: la guarda el Chat y la respetan Documentos,
el guion de Vídeo, el Cuaderno, Comprensión y la barra de estado
(`services/chat_engine.py`). Además de la nube, el mismo motor sirve para
**servidores locales OpenAI-compatible** (`llama_cpp` → `llama-server` en
`127.0.0.1:8080`, `lm_studio` en `127.0.0.1:1234`), sin clave. En esos servidores
locales la app **consulta `GET {base_url}/models`** y añade a los selectores **los
modelos que sirve de verdad** el servidor (p. ej. el `.gguf` que cargaste con
`llama-server`), sin escribirlos en `engines.json`. También hay **imagen externa**
(`openai_image`, p. ej. `gpt-image-1`) como motor de la pestaña Imagen.

## Retoque de fotos

Pestaña **Retoque** (al lado de Imagen): abre imágenes en **cualquier formato**,
aplica ajustes de **brillo/contraste/color/nitidez** (Pillow, funcionan sin
instalar nada), mejora con **IA** (Real-ESRGAN x2/x3/x4 y restauración de caras
con GFPGAN) y guarda en PNG/JPEG/WEBP/TIFF/BMP.

La IA va en un venv aparte para no arrastrar torch al arranque de la app:

```powershell
py -3.12 -m venv venv-photo
venv-photo\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
venv-photo\Scripts\python.exe -m pip install realesrgan gfpgan
```

La primera mejora descarga los pesos: Real-ESRGAN ~64 MB; GFPGAN ~330 MB la
primera vez que marcas «Restaurar caras». Si `basicsr` falla al importar por
`functional_tensor`, parchea ese import a `torchvision.transforms.functional`.

## Retoque de vídeo

Pestaña **Vídeo IA** (tras Vídeo): **reescala**, ajusta el **color**, cambia los
**fotogramas** y la **velocidad** de un vídeo.

- **Color/efectos (ffmpeg)**: brillo, contraste, saturación, gamma, reduce-ruido
  (`hqdn3d`), nitidez (`unsharp`) e interpolación de FPS (`minterpolate`).
  Instantáneo y sin instalar nada.
- **Cámara lenta y repeticiones**: velocidad de **0.25x a 2x** sobre todo el vídeo
  y *replay* de un momento (del segundo *A* al *B*, repetido hasta 6 veces, con la
  **última vuelta en cámara lenta**). Pensado para un partido: volver a ver el gol
  o la parada con el momento clave ya resuelto.
  - El **audio va pegado al vídeo**: se reajusta con `atempo`, así que a 0.5x el
    sonido también va a la mitad y no se desincroniza. Si el vídeo es mudo, se
    monta sin audio (no se le exige una pista que no tiene).
  - Con **«Suavizar el movimiento»** se interpolan los fotogramas que faltan
    (`minterpolate`) antes de estirar el tiempo: la cámara lenta se ve continua en
    vez de a saltos. Cuesta bastante más de procesar, así que se puede desactivar.
  - Todo se monta en **un solo pase** de ffmpeg: no quedan ficheros intermedios.
- **Rampa de velocidad**: ralentiza **sólo** el tramo elegido y deja el resto a
  velocidad normal (la rampa de las retransmisiones: llega la jugada y baja la
  velocidad). Con **Zoom** (1x–3x) la repetición se ve de más cerca; el zoom
  devuelve la misma resolución que el original, para no romper un montaje después.
- **Rótulos**: escribe una línea por rótulo (`0:12 Gol de Álex`) y se queman sobre
  el vídeo con el estilo de subtítulos del Ensamblador (posición, tamaño, color).
  Sirve para el marcador, el minuto o el nombre del jugador.
- **Enviar al Ensamblador**: manda a esa pestaña lo último montado (replay, rampa,
  rótulos o cámara lenta) para añadirle música, voz o subtítulos.
- **Comprimir (recodificar más pequeño)**: baja el peso **sin tocar la imagen**, con
  presets de calidad (CRF en CPU / CQ en GPU, ya compensados para dar un tamaño
  parecido) y codificador **CPU (libx264)** o **GPU (NVENC, más rápido)**. El audio
  se pasa a AAC a 192 k. Medido con un trozo de cámara real (1080p50, 25,4 Mbps,
  34 s): «Equilibrada» deja **108 MB → 35 MB (8,3 Mbps) en 6,4 s** con CPU, o
  **39 MB en 4,9 s** con GPU; «Mínima», 13-16 MB. Para el partido completo (52 min),
  eso son unos 10 minutos de proceso y ~3 GB de archivo. El resultado va a
  `projects/grabaciones/`.
- **Unir grabaciones troceadas**: las cámaras parten la grabación en `00000.MTS`,
  `00001.MTS`… Eliges los trozos (se ordenan por nombre), los unes en **un solo
  MP4** y ya se pueden usar en cualquier pestaña. El **vídeo se copia** (sin
  recodificar y sin perder calidad: 2 GB tardan segundos) y sólo el audio pasa a
  AAC, porque las cámaras graban AC-3 y así lo abre cualquier reproductor. La
  unión sale a `projects/grabaciones/`, fuera de la retención de `output/`.
  - **Igualar los trozos (recomprimir)**: márcalo si los trozos vienen ya
    retocados (una rampa, una repetición o una cámara lenta cambian el ritmo; un
    efecto de velocidad además deja el audio en AAC y la cámara graba AC-3, y el
    demuxer `concat` no puede pegar códecs distintos). Entonces se normaliza cada
    trozo —mismo tamaño, mismo ritmo estándar y audio a estéreo— con el filtro
    `concat` antes de pegarlos. Cuesta una recodificación: **medido, 11 min de
    1080p50 en 66 s**. Sin marcar, la app intenta primero la copia (segundos) y
    sólo recompone si falla.
- **Escalado con IA (Real-ESRGAN)**: extrae los fotogramas, los reescala con
  `venv-photo` (fotograma a fotograma, cargando el modelo una vez) y vuelve a
  montar el vídeo con el audio original.

Para la música de fondo sin MusicGen: deja tus MP3/WAV en `music/` y el Ensamblador
los mezcla con *ducking* bajo la voz.

Reutiliza lo ya instalado para el Retoque de fotos (`venv-photo`): no hay que
instalar nada nuevo. Filtros y grafo del replay en
[`services/video_enhance_params.py`](services/video_enhance_params.py).

## Rendimiento

- **Razonamiento apagado donde estorba.** Los `qwen3*` (chat y visión) gastan
  ~90 % de los tokens en `thinking` y, con la ventana llena, devuelven vacío
  («se agotó el espacio de la respuesta»). Los motores `ollama` y
  `ollama_vision` admiten en `engines.json` `"think": false` (apaga el
  razonamiento y añade `/no_think` al prompt); la visión admite además
  `"num_predict": 1024` (acota la salida). El modelo de visión por defecto es
  `qwen2.5vl:7b` (**sin** razonamiento): los `qwen3-vl` por fotograma gastan el
  presupuesto de salida «pensando» y el Cuaderno falla con «se agotó la ventana de
  contexto»; si usas un `qwen3-vl`, quita `num_predict` y sube `num_ctx`.
- **Caché de fotogramas.** Comprensión y Cuaderno extraen fotogramas con ffmpeg;
  [`core/frame_cache.py`](core/frame_cache.py) los guarda en `cache/frames/`
  (indexados por vídeo y tiempo) para no volver a decodificar los mismos entre
  análisis o reintentos. Se puede podar con `FrameCache().prune(max_files=...)`.

## Cómo arrancar

```powershell
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Para reproducir **las versiones exactas** con las que se probó esta copia, añade
el fichero de restricciones: una actualización de `torch`, `onnxruntime` o
`ctranslate2` (los arrastran Kokoro, Piper y Whisper) puede romper un motor sin
haber tocado una línea del proyecto.

```powershell
pip install -r requirements.txt -c constraints.txt
```

`venv\Scripts\python.exe -m pip freeze > constraints.txt` lo regenera después de
actualizar dependencias.

Para uso diario en Windows, usa `Imperator.lnk` (acceso directo con el icono y el
`AppUserModelID`): arranca con `pythonw.exe` (sin ventana de consola) y deja en
la barra de tareas el icono de Imperator. Ancla ese acceso directo (clic derecho
→ "Anclar a la barra de tareas") para que quede fijo.

Si falta `Imperator.lnk`, regenera el icono y el acceso directo con:

```powershell
python tools/make_icon.py        # assets/icono.ico en DIB (necesita Pillow)
python tools/create_shortcut.py  # Imperator.lnk (necesita comtypes)
```

Alternativa rápida sin icono personalizado: doble clic en `Imperator.bat`.

## Tests y lint

Sin dependencias de runtime (stdlib `unittest`):

```powershell
python -m compileall -q .
python -m ruff check .                      # requiere: pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

El CI (`.github/workflows/ci.yml`) ejecuta los tres pasos, instala
`requirements-test.txt` con las versiones congeladas de `constraints-test.txt`
(incluida la de `ruff`) y activa el smoke test de la GUI.

Las herramientas de desarrollo viven en `requirements-dev.txt` (`ruff` y
`pyright`): no hacen falta para ejecutar ni para la suite, y `pyright` queda fuera
del CI a propósito (necesita Node y con PyQt6 da más ruido que señal; ver TODO.md
§0). Donde más informa es en las capas sin Qt: `pyright core services`.

## Configuración y secretos

- La configuración vive en `config/*.json` (ignorada por git); las plantillas
  versionadas son `config/*.example.json`.
- Las URL de los motores locales se escriben con **`127.0.0.1`, no `localhost`**:
  en Windows `localhost` intenta primero IPv6 (`::1`), donde Ollama y ComfyUI no
  escuchan, y cada petición pagaba ~2 s de fallback (medido: 2 040 ms frente a
  15 ms). Si tu `engines.json` venía con `localhost`, la app lo avisa en el log.
- Los secretos se leen **solo** de `.env` (p. ej. `BFL_API_KEY`); el campo
  `key` de `apis.json` está deprecado y se ignora.
- `settings.json` acepta `log_level` y `rutas.logs` para el nivel de log y un
  log rotado a diario (`<logs>/plataformaia.log`), además de `gpu_scheduler`
  (prioridades y reserva de VRAM; ver ARCHITECTURE.md §9) y `output_limpieza`
  (retención de lo generado en `output/`: por defecto 500 ficheros, 90 días y
  8 GB, con 24 h de gracia para lo recién generado; se aplica al arrancar y sólo
  mira ficheros con extensión de la app).
- Los motores de percepción (`yolo`, `clip`, `paddle_ocr`) son **opcionales** y
  vienen deshabilitados: sin sus dependencias (`ultralytics`, `transformers` +
  `torch`, `paddleocr`) la app arranca igual y el motor aparece como no
  disponible. Se activan en *Motores → Gestionar motores…*.
- Las librerías pesadas de los motores (torch/transformers/kokoro) se importan
  **al usar el motor**, no al arrancar: la ventana queda lista en ~0,5 s en vez de
  ~12 s. La primera vez que uses Kokoro en una sesión esa carga (~10 s) se paga
  ahí, no en el arranque de todo el mundo.
