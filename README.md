# Imperator (PlataformaIA)

Plataforma modular de IA con backend Python y GUI PyQt6: unifica motores locales (Ollama, llama.cpp, LM Studio, ...), modelos (LLM, imagen, vídeo, audio, multimodal), APIs externas (OpenAI, DeepSeek, Kling, ...) y tareas tipo plugin (chat, OCR, imagen, vídeo, TTS/STT, documentos).

## Cómo arrancar

```powershell
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

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

## Tests

Sin dependencias de runtime (stdlib `unittest`):

```powershell
python -m compileall -q .
python -m unittest discover -s tests -v
```

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
  (prioridades y reserva de VRAM; ver ARCHITECTURE.md §9).
- Los motores de percepción (`yolo`, `clip`, `paddle_ocr`) son **opcionales** y
  vienen deshabilitados: sin sus dependencias (`ultralytics`, `transformers` +
  `torch`, `paddleocr`) la app arranca igual y el motor aparece como no
  disponible. Se activan en *Motores → Gestionar motores…*.
