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

**Estado:** no iniciada — motor opcional.

- Añadir un motor local `engines/musicgen.py` que genere música con
  [Meta MusicGen](https://github.com/facebookresearch/audiocraft).
- Seguir el patrón de `engines/base_engine.py` (`Engine`: `id`, `nombre`,
  `is_available()`, `list_models()`).
- **Ojo:** el ABC `Engine` actual está orientado a texto (obliga a `chat()`).
  En `base_engine.py` hay un `TODO(fase N)` para separar
  `Engine` / `ImageEngine` / `VideoEngine` / `AudioEngine`. Convendría crear
  un `MusicEngine` (o `AudioEngine`) con `generate_music(...)` en lugar de
  forzar `chat()`.
- Registrar el motor en `core/engine_manager.py` y exponerlo en la UI de motores.

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
3. `TODO(fase 6)` de `core/task_manager.py`: dependencias de tareas por `kwargs`.
4. Los MP3 en `music/` para validar el ducking (bloqueado por material de prueba).
5. Fase opcional: motor local MusicGen.
