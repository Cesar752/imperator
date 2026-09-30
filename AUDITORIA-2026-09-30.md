# Auditoría técnica — Imperator (PlataformaIA)

**Fecha:** 30/09/2026 · **Alcance:** `C:\Imperator` (excluye `venv/`, `.git/`, `output/`, `voices/`, `music/`, `assets/`)
**Método:** revisión de código y configuración, ejecución de las verificaciones del propio proyecto (ruff, compileall, unittest), inspección de git y búsqueda dirigida de patrones de riesgo.

---

## 1. Resumen ejecutivo

El proyecto está en **buen estado**: las verificaciones automáticas pasan limpias, el diseño es coherente con lo documentado, la deuda técnica está declarada con mediciones reales y las prácticas de seguridad básicas (secretos, subprocess, timeouts, escritura atómica de config) están bien resueltas.

No se han encontrado **fallos críticos de código**. Los hallazgos relevantes son de tres tipos: una credencial de pago real en texto plano en disco, la ausencia de lockfile de runtime y varios desajustes entre la documentación y el código/configuración que, en un proyecto que se apoya tanto en su documentación, son el riesgo más probable de errores futuros.

| Severidad | Nº | Naturaleza | Estado |
| --------- | -- | ---------- | ------ |
| Alta | 1 | Credencial de API de pago real en `.env` en claro | **Pendiente** (lo asume el autor) |
| Media | 4 | Lockfile de runtime, cobertura de tests, drift docs↔código, estado compartido entre hilos | **Resueltos** (§7) |
| Baja | 6 | `.lnk` versionado, artefactos huérfanos, tipado fuera de CI, saneado de entrada, ruta muerta, log a fichero desactivado | **Resueltos** (§7) |

---

## 2. Verificaciones ejecutadas (evidencia)

| Comprobación | Comando | Resultado |
| ------------ | ------- | --------- |
| Lint | `venv\Scripts\python.exe -m ruff check .` | **All checks passed!** |
| Compilación | `python -m compileall -q .` | **OK** (árbol completo) |
| Tests | `IMPERATOR_GUI_SMOKE=1 python -m unittest discover -s tests` | **Ran 693 tests — OK (skipped=2)** en 2,6 s (631 al auditar; +62 en §7) |
| Git | `git status --short` | **Árbol limpio** al auditar; 21 tags; último commit `1856278` (hoy) |
| Historial de secretos | `git log --all -- .env` | **Vacío** (`.env` nunca se versionó) |
| Ficheros versionados | `git ls-files` | **122** al auditar (**130** tras §7: +2 de dependencias, +6 de tests, +1 este informe, −1 `Imperator.lnk`); solo `*.example.json` de config |
| Patrones peligrosos | `shell=True`, `os.system`, `eval`, `exec`, `pickle`, `yaml.load` | **Ninguno** |
| `except Exception` sin `noqa` | grep dirigido | **Ninguno** |
| `print()` en código de app | grep dirigido | **Ninguno** (solo en `tools/`) |

**Cobertura:** no hay herramienta de cobertura configurada, por lo que la cobertura real es **desconocida**. La ratio tests/LOC es alta (631 tests sobre ~9 000 líneas de app), pero desigual por módulo (§4.2).

---

## 3. Hallazgos

### 3.1 [ALTA] Credencial de pago real en texto plano

`C:\Imperator\.env` contiene una clave **real** de Black Forest Labs (`BFL_API_KEY`, 37 caracteres, prefijo `bfl_`). BFL es una API de pago: la clave tiene coste económico directo.

**Lo que sí está bien:** `.env` está en [`.gitignore`](../../../Imperator/.gitignore:19) y el historial de git está limpio (`git log --all -- .env` no devuelve nada); el código ignora deliberadamente el campo `key` de `apis.json` y lee siempre del entorno ([`core/config_manager.py`](../../../Imperator/core/config_manager.py:391)); el motor BFL no registra la clave en logs ([`engines/bfl.py`](../../../Imperator/engines/bfl.py:82)).

**Riesgo residual:** cualquier copia de seguridad, sincronización de la carpeta (OneDrive/Drive), indexado de disco o captura de pantalla del explorador expone una credencial facturable. El `.env` no se borra al desinstalar y no hay rotación.

**Recomendaciones:**

1. Confirmar que `C:\Imperator` no está dentro de una carpeta sincronizada en la nube.
2. Verificar permisos NTFS: solo la cuenta del usuario, sin herencia de `Users`.
3. Rotar la clave en <https://docs.bfl.ai> si hubo cualquier duda de exposición.
4. A medio plazo, mover la clave de pago a Credential Manager (DPAPI) en lugar de `.env`; el `ConfigManager` ya tiene el punto único de resolución para hacerlo sin tocar los motores.

---

### 3.2 [MEDIA — resuelto] Sin lockfile para el entorno de runtime

[`requirements.txt`](../../../Imperator/requirements.txt:1) declara rangos (`PyQt6>=6.6.0,<7`, `numpy>=1.26.0,<3`, `kokoro>=0.9.4,<1`, …) y **no existe** un `constraints.txt` de runtime. Solo el entorno de test está congelado en [`constraints-test.txt`](../../../Imperator/constraints-test.txt:1), que **no cubre** las dependencias pesadas que realmente ejecutan los motores: `torch`/`transformers` (vía `kokoro`), `ctranslate2` (vía `faster-whisper`), `onnxruntime` (vía `piper-tts`), `PyMuPDF`.

**Impacto:** una reinstalación o actualización puede traer versiones incompatibles de torch/onnxruntime y romper motores sin que nadie haya tocado el proyecto. El propio README ya recomienda generar `constraints.txt` y nunca se hizo; el CI queda protegido, el usuario no.

**Recomendación:** `venv\Scripts\python.exe -m pip freeze > constraints.txt` y documentar `pip install -r requirements.txt -c constraints.txt` en el README.

---

### 3.3 [MEDIA — resuelto] Cobertura de tests desigual en los módulos que más escriben en disco

Existen 631 tests, pero **no tienen test** (dedicado ni indirecto):

- [`engines/piper_tts.py`](../../../Imperator/engines/piper_tts.py:26) — escribe WAV, cachea modelos ONNX.
- [`engines/kokoro_tts.py`](../../../Imperator/engines/kokoro_tts.py:1) — importa torch de forma perezosa.
- [`engines/whisper_engine.py`](../../../Imperator/engines/whisper_engine.py:1) — carga CTranslate2 y escribe SRT.
- [`core/frame_extractor.py`](../../../Imperator/core/frame_extractor.py:1) — invoca ffmpeg (cubierto solo de refilón desde `test_video_understanding`).
- [`ui/engines_dialog.py`](../../../Imperator/ui/engines_dialog.py:1) — CRUD de motores (validación de entradas).
- La mayoría de widgets: [`ui/video_widget.py`](../../../Imperator/ui/video_widget.py:1) (606 líneas), [`ui/assembler_widget.py`](../../../Imperator/ui/assembler_widget.py:1) (562), [`ui/understanding_widget.py`](../../../Imperator/ui/understanding_widget.py:1) (487), [`ui/image_widget.py`](../../../Imperator/ui/image_widget.py:1) (376), [`ui/audio_widget.py`](../../../Imperator/ui/audio_widget.py:1) (285) y [`ui/main_window.py`](../../../Imperator/ui/main_window.py:1) (446, solo smoke test).

Los tres motores sin test son justo los que cargan modelos pesados y generan ficheros: la parte con más efectos secundarios es la que menos red tiene. Los widgets restantes son candidatos naturales a tests de humo por pestaña (construir, validar parámetros, cancelar), en la línea de los ya existentes `test_chat_widget`/`test_document_widget`.

---

### 3.4 [MEDIA — resuelto] Desajustes documentación ↔ código/configuración

En un proyecto cuya operativa depende de sus notas, un desajuste es un error futuro garantizado. Detectados:

| Documento | Dice | Realidad |
| --------- | ---- | -------- |
| [`ARCHITECTURE.md`](../../../Imperator/ARCHITECTURE.md:370) §8 | `MAX_PARALLEL_PAGES = 2`, visión `num_ctx` 32768 y chat 16384 | [`ui/document_widget.py`](../../../Imperator/ui/document_widget.py:66) define `MAX_PARALLEL_PAGES = 1`; [`config/engines.json`](../../../Imperator/config/engines.json:1) usa 8192/16384. El propio documento se contradice más abajo (línea 446: «`MAX_PARALLEL_PAGES` vuelve a `1`») |
| [`ARCHITECTURE.md`](../../../Imperator/ARCHITECTURE.md:321) §8 | «620 tests» | 631 |
| [`TODO.md`](../../../Imperator/TODO.md:29) §0 / §7 | «596 tests» y «573 tests» | 631 (cifras de sesiones distintas, sin marcar como históricas) |
| [`TODO.md`](../../../Imperator/TODO.md:114) §1 | «`music/` está vacío (0 archivos)» | `music/` tiene 3 MP3: el pendiente de validar el ducking ya se puede cerrar |
| [`config/engines.json`](../../../Imperator/config/engines.json:1) | — | **No contiene `musicgen`**, pero [`DEFAULT_ENGINES`](../../../Imperator/core/config_manager.py:171) lo define con `enabled: true`. Si se borra `engines.json`, el motor reaparece: la configuración real y los defaults/plantilla divergen |
| [`PLAN.md`](../../../Imperator/PLAN.md:55) | Carpetas `apis/`, motores `llama_cpp`/`lmstudio` | No existen (el propio PLAN se declara parcialmente obsoleto) |

**Recomendación:** actualizar §8 de ARCHITECTURE (o convertirlo en un histórico fechado como TODO.md) y alinear `config/engines.json` con la plantilla (`musicgen`).

---

### 3.5 [MEDIA — resuelto] Estado mutable compartido entre hilos en los motores

Los motores son **instancias únicas** que `EngineManager` comparte entre todos los workers (QThread) y el hilo de UI. Dos puntos escriben estado sin sincronización:

- [`engines/ollama.py`](../../../Imperator/engines/ollama.py:106): `chat()` asigna `self._last_used_model = model`. `unload_model()` lo lee después, y puede invocarse desde el hilo de UI (`release_engines` → *Liberar VRAM ahora*) mientras un worker está en `chat()`.
- [`engines/piper_tts.py`](../../../Imperator/engines/piper_tts.py:42): `self._loaded` es un `dict` cache de voces sin lock; dos síntesis concurrentes pueden cargar el mismo ONNX dos veces (o, en el peor caso, un lector ver un dict a medias).

**Impacto:** bajo (valores benignos, sin corrupción de datos), pero es el **único punto del diseño sin protección**, cuando el resto sí la usa: `ConfigManager` con `RLock` ([`core/config_manager.py`](../../../Imperator/core/config_manager.py:323)), escrituras atómicas ([línea 461](../../../Imperator/core/config_manager.py:461)) y `GpuScheduler` declarado thread-safe.

**Recomendación:** o bien documentar el contrato («una tarea pesada a la vez» ya lo garantiza `gpu_slot` con `max_concurrent=1`), o bien un `Lock` mínimo en el cache de voces y hacer `_last_used_model` de solo lectura para la UI.

---

### 3.6 [BAJA — resuelto] `Imperator.lnk` versionado en git

`git ls-files` incluye `Imperator.lnk`, un binario con **rutas absolutas de esta máquina** (`pythonw.exe` del venv, `C:\Imperator\...`). No es portable, cambia con cada regeneración y ensucia los diffs. El README ya explica cómo regenerarlo ([`tools/create_shortcut.py`](../../../Imperator/tools/create_shortcut.py:1)).

**Recomendación:** añadir `*.lnk` a [`.gitignore`](../../../Imperator/.gitignore:1) y `git rm --cached Imperator.lnk`; dejar `Imperator.bat` (texto, portable) como el artefacto versionado.

### 3.7 [BAJA — resuelto] Artefactos huérfanos en `__pycache__`

`C:\Imperator\__pycache__\` contiene `_min_test.cpython-312.pyc` y `_min_test2.cpython-312.pyc` **sin fuente `.py`**. Son restos de pruebas manuales; no afectan a la ejecución (está ignorado por git) pero ensucian y despistan.

### 3.8 [BAJA — resuelto] Sin verificación de tipos en CI

`pyright` está configurado ([`pyrightconfig.json`](../../../Imperator/pyrightconfig.json:1)) pero **fuera del CI**, con la justificación (razonable y documentada en TODO §0) de que en modo `basic` con PyQt6 da más ruido que señal. Efecto colateral: `pyright` no está en `requirements-test.txt`, así que un colaborador nuevo no lo tiene disponible sin buscarlo. Alternativa de bajo coste: añadir `mypy`/`pyright` **solo** a `core/` y `services/` (lógica pura sin Qt), donde el tipado sí sería informativo.

### 3.9 [BAJA — resuelto] Nombre de voz sin sanear en la ruta de entrada (Piper)

[`engines/piper_tts.py`](../../../Imperator/engines/piper_tts.py:95) construye `self._voices_dir / f"{voice_name}.onnx"` sin comprobar que `voice_name` sea un stem real de `voices/`. El saneado existe para el **nombre de salida** ([línea 117](../../../Imperator/engines/piper_tts.py:117)) pero no para la **entrada**. La UI solo ofrece voces listadas, así que el vector requiere editar `settings.json` a mano (`last_models.audio_voice_piper`). Riesgo bajo (app local, sin privilegios), fácil de cerrar validando contra `list_voices()`.

### 3.10 [BAJA — resuelto] Referencia muerta tras la retención de `output/`

`settings.json` → `last_models.video_last_image` apunta a un fichero de `output/`. La retención ([`services/output_cleanup.py`](../../../Imperator/services/output_cleanup.py:161)) puede borrarlo (p. ej. a los 90 días) y la pestaña Vídeo restaurará una ruta inexistente. Cosmético: conviene `Path.exists()` al restaurar.

### 3.11 [BAJA — resuelto] Log a fichero desactivado

`config/settings.json` tiene `rutas: {}`, así que no se escribe `logs/plataformaia.log`. En una app de escritorio con fallos intermitentes de motores locales (OOM, timeouts de Ollama, modelos no cargados), no hay traza persistente para diagnosticar a posteriori. Activarlo (`"logs": "logs"`) tiene coste cero.

### 3.12 [BAJA — resuelto] Doble pin de ruff

`ruff==0.16.9` está fijado en el CI ([`.github/workflows/ci.yml`](../../../Imperator/.github/workflows/ci.yml:1)) **y** el rango de reglas en [`ruff.toml`](../../../Imperator/ruff.toml:13). Es correcto, pero son dos sitios que mantener sincronizados con la versión del propio entorno (`.ruff_cache/0.16.9` sugiere que coinciden hoy).

---

## 4. Fortalezas verificadas (no solo declaradas)

- **Higiene de subprocess:** sin `shell=True` en todo el árbol; todos los `subprocess.run` pasan lista de argumentos y `timeout`: ffmpeg en [`core/video_assembler.py`](../../../Imperator/core/video_assembler.py:38) y [`core/frame_extractor.py`](../../../Imperator/core/frame_extractor.py:96), `nvidia-smi` en [`core/gpu_scheduler.py`](../../../Imperator/core/gpu_scheduler.py:132). Un binario atascado no cuelga el worker.
- **Red:** **todas** las llamadas `requests` llevan `timeout` explícito (verificado motor por motor); `raise_for_status()` se traduce a `EngineError` con mensaje accionable; sin `verify=False`.
- **Descargas acotadas:** [`engines/base_engine.py`](../../../Imperator/engines/base_engine.py:29) define `MAX_DOWNLOAD_BYTES` (512 MiB) y valida el `Content-Length` declarado **y** el cuerpo recibido antes de escribir.
- **Secretos:** `.env` ignorado y nunca commiteado; `resolve_api_key` ignora el campo `key` deprecado; las plantillas `*.example.json` no contienen credenciales.
- **Configuración robusta:** validación por esquema con tipos/`allowed`/`min_len`, detección de ids duplicados case-insensitive, escritura atómica con `mkstemp` + `os.replace` y `RLock`, y carga tolerante a fallos (un JSON corrupto no impide arrancar; [`main.py`](../../../Imperator/main.py:30)).
- **Operaciones destructivas prudentes:** la retención de `output/` usa lista blanca de extensiones, ventana de gracia de 24 h y aplica la política solo a productos de la app (`services/output_cleanup.py`), nunca lanza y avisa en el log.
- **CI sólido:** `permissions: contents: read`, versiones de test congeladas, lint antes de los tests y smoke test de GUI activado (`IMPERATOR_GUI_SMOKE=1`).
- **Rendimiento y arranque:** imports perezosos de `kokoro`/`perception` con `find_spec` para `is_available()`, protegidos por [`tests/test_startup_imports.py`](../../../Imperator/tests/test_startup_imports.py:1) — un test-guarda contra la regresión más cara (11,9 s → 0,5 s de arranque).
- **Deuda técnica honesta:** TODO.md registra decisiones descartadas **con mediciones** y el motivo; es la razón principal por la que el drift de §3.4 se detecta rápido.

---

## 5. Plan de acción priorizado

| # | Acción | Severidad | Coste |
| - | ------ | --------- | ----- |
| 1 | Verificar que `.env` no está en carpeta sincronizada; revisar ACL NTFS; rotar la clave BFL si hubo duda | Alta | 15 min |
| 2 | Generar `constraints.txt` de runtime y documentar el `pip install -c` | Media | 10 min |
| 3 | Actualizar `ARCHITECTURE.md` §8 (nº de tests, `MAX_PARALLEL_PAGES`, `num_ctx`) y marcar TODO §1 como resuelto (`music/` ya tiene MP3) | Media | 20 min |
| 4 | Alinear `config/engines.json` con la plantilla (añadir `musicgen`) | Media | 5 min |
| 5 | Test de humo por widget restante (video, assemblador, comprensión, imagen, audio) y tests de `piper_tts`/`kokoro_tts`/`whisper_engine` con dobles | Media | 2-4 h |
| 6 | `*.lnk` a `.gitignore` + `git rm --cached Imperator.lnk` | Baja | 5 min |
| 7 | Borrar `__pycache__` de la raíz; activar `rutas.logs` en `settings.json`; validar `voice_name` contra `list_voices()`; `Path.exists()` al restaurar `video_last_image` | Baja | 20 min |
| 8 | Documentar el contrato de concurrencia de motores o añadir `Lock` al cache de voces | Baja | 30 min |

---

## 6. Conclusión

Imperator es un proyecto **maduro y bien auditado internamente**: las verificaciones automáticas pasan, la arquitectura se corresponde con el código en lo esencial y las áreas de riesgo clásicas (ejecución de comandos, red sin timeout, secretos en repositorio, escrituras concurrentes de configuración) están resueltas con criterio y con tests.

La brecha real no está en el código, sino en la **sincronización entre lo que el proyecto dice de sí mismo y lo que realmente hace** (§3.4): cuando la documentación es la fuente de verdad operativa, cada número desactualizado es una decisión futura mal informada.

---

## 7. Estado tras las mejoras (30/09/2026)

Se cerraron el lockfile de runtime y los **nueve hallazgos de severidad media y baja**. El de severidad alta (§3.1) queda abierto por decisión del autor.

| Hallazgo | Qué se hizo |
| -------- | ----------- |
| 3.2 | `constraints.txt` con el `pip freeze` del venv de referencia (204 paquetes) y el `pip install -r requirements.txt -c constraints.txt` documentado en el README |
| 3.3 | **62 tests nuevos** (`test_piper_tts` 13, `test_kokoro_tts` 9, `test_whisper_engine` 12, `test_frame_extractor` 12, `test_engines_dialog` 10, `test_widgets_smoke` 6), todos con dobles: no necesitan motores, GPU, red ni ffmpeg |
| 3.4 | `ARCHITECTURE.md` §8 corregido (`MAX_PARALLEL_PAGES = 1` y `num_ctx` 8192/16384 reales, nº de tests ajustado y referencia a este informe), `TODO.md` §1 deja de dar `music/` por vacío y `config/engines.json` recupera `musicgen` |
| 3.5 | Lock en `PiperTTSEngine._loaded` y en `OllamaEngine._last_used_model`; el porqué queda escrito en el código |
| 3.6 | `*.lnk` añadido a `.gitignore` y `git rm --cached Imperator.lnk` |
| 3.7 | `__pycache__/` de la raíz borrado (los `_min_test*` sin fuente) |
| 3.8 | `requirements-dev.txt` (`ruff` + `pyright`) y nota en el README sobre cuándo usarlo |
| 3.9 | El nombre de voz de Piper se valida contra `list_voices()` antes de construir la ruta |
| 3.10 | La última imagen de la pestaña Vídeo se restaura solo si el fichero sigue existiendo |
| 3.11 | `settings.json` → `rutas.logs = "logs"`: log rotado a diario activado |
| 3.12 | El pin de `ruff` vive solo en `constraints-test.txt`; el workflow hace `pip install ruff -c constraints-test.txt` |

**Verificación final:** `ruff check .` limpio, `compileall` OK y **693 tests en verde** (2 skips) con `IMPERATOR_GUI_SMOKE=1`. Detalle de la sesión en `TODO.md` §8.
