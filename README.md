# Imperator (PlataformaIA)

Plataforma modular de IA con backend Python y GUI PyQt6: unifica motores locales (Ollama, llama.cpp, LM Studio, ...), modelos (LLM, imagen, vídeo, audio, multimodal), APIs externas (OpenAI, DeepSeek, Kling, ...) y tareas tipo plugin (chat, OCR, imagen, vídeo, TTS/STT, documentos).

## Cómo arrancar

```powershell
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Tests

Sin dependencias de runtime (stdlib `unittest`):

```powershell
python -m compileall -q .
python -m unittest discover -s tests -v
```

## Configuración y secretos

- La configuración vive en `config/*.json` (ignorada por git); las plantillas
  versionadas son `config/*.example.json`.
- Los secretos se leen **solo** de `.env` (p. ej. `BFL_API_KEY`); el campo
  `key` de `apis.json` está deprecado y se ignora.
- `settings.json` acepta `log_level` y `rutas.logs` para el nivel de log y un
  log rotado a diario (`<logs>/plataformaia.log`).
