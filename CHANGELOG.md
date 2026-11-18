# Changelog

Formato: [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/).
Versionado: [SemVer](https://semver.org/lang/es/).

## [Unreleased]

### Added
- `pyproject.toml` con metadatos y dependencias (antes solo en `requirements*.txt`).
- Spec de PyInstaller para preparar el empaquetado (sin construir todavía).
- Receptor `QObject` (`ui/pool_callbacks.py`) para callbacks del pool.

### Changed
- `core/task_manager.py` importa las tareas a demanda: importar el núcleo ya no
  carga PyQt6.
- `EngineManager._build` pasa de una cadena de 172 líneas a un mapa
  `id -> builder`.
- `_check_cancel` vive solo en `ThreadWorker` (se eliminan 4 copias).
- Colores semánticos centralizados en `ui/theme.py` (`SEMANTIC`), con un valor
  por tema.
- `_apply_no_think`, `is_loaded` y `unload_model` de Ollama, una sola definición
  en `_OllamaCommon`.
- Escritura atómica de JSON compartida (`services/atomic_io.py`).

### Fixed
- Arranque robusto si `settings.json` está corrupto o con BOM (antes moría tras
  el splash).
- Aborto del proceso por callbacks del pool sobre widgets destruidos (C2).
- `PoolTask` retenido hasta que termina su hilo (evita emitir sobre señales
  muertas).
- `is_available()` y la descarga de VRAM ya no bloquean el hilo de UI; miniaturas
  de escena, conversión MP3 y reproceso de foto van al pool.
- Preferencias: una lectura de `settings.json`, no una por valor (de 11 a 1).
- `excepthook` global que registra en `logs/crash.log` y no abre diálogo en tests.
- Timeout en las llamadas a ffmpeg de `jarvis_bridge` (antes podía colgarse).
- EOL: `ruff` limpio (2 E402 en `jarvis_bridge`).

### Removed
- Artefactos del índice de git (venvs, caches, PDF del manual).
