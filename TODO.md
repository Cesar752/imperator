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
