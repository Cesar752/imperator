# PLAN — PlataformaIA

## Objetivo del proyecto

PlataformaIA es una plataforma modular de inteligencia artificial con backend en Python y una interfaz gráfica de escritorio construida con PyQt6. Su propósito es unificar en un único punto de trabajo todos los motores de IA que un usuario tenga instalados localmente (Ollama, llama.cpp, LM Studio, exllamav2, vllm, TensorRT-LLM, etc.), los distintos tipos de modelos que estos exponen (LLM, imagen, vídeo, audio y multimodal) y las APIs externas de proveedores en la nube (OpenAI, DeepSeek, Kling, etc.).

El objetivo final es que el usuario pueda elegir libremente qué motor o API ejecuta cada tarea, sin atarse a un proveedor ni a una herramienta concreta. Para ello, la plataforma expone "tareas" de alto nivel —chat, OCR, generación de imagen, generación de vídeo, síntesis y reconocimiento de voz (TTS/STT) y lectura de documentos— que funcionan como plugins independientes y desacoplados del motor subyacente.

La arquitectura está pensada para crecer de forma incremental y segura: cada motor, cada API y cada tarea se registran siguiendo una interfaz común, de modo que añadir soporte para algo nuevo no implica modificar el núcleo, sino añadir un módulo nuevo. Los secretos (claves de API) nunca viven en el código: se leen desde variables de entorno (`.env`) o desde archivos de configuración ignorados por git.

## Decisiones de arquitectura

### Por qué PyQt6
- Es la opción de referencia para GUIs de escritorio en Python, con soporte a largo plazo y buena integración con el ecosistema (señales/slots, hilos con `QThread` para no bloquear la UI durante inferencias largas).
- Permite una UI nativa y responsiva sin depender de un navegador ni de un servidor web, lo que encaja con una herramienta de escritorio local que habla con motores locales y APIs remotas.
- Su modelo de eventos (signals/slots) facilita que las tareas asíncronas (generación de imagen/vídeo, streaming de chat) actualicen la interfaz de forma segura.

### Por qué JSON para la configuración
- Es legible, editable a mano y sin dependencias externas (módulo `json` de la stdlib).
- Permite versionar plantillas (`*.example.json`) y mantener fuera de git las reales, que pueden contener secretos o rutas locales.
- Es suficientemente expresivo para describir motores, modelos, APIs y tareas, y fácil de validar con un esquema propio en `config_manager.py`.

### Por qué plugins para las tareas
- Las tareas (chat, OCR, imagen, vídeo, audio, documentos) son muy heterogéneas: cada una tiene entradas, salidas, modelos y motores distintos.
- Un sistema de plugins permite añadir o quitar tareas sin tocar el núcleo, reutilizar la infraestructura común (selección de motor/API, configuración, logging) y mantener cada tarea en un fichero aislado y testeable.
- El gestor de tareas (`task_manager.py`) descubre y registra los plugins dinámicamente, igual que el gestor de motores y el de APIs.

## Estructura de carpetas

```
PlataformaIA/            (raíz = C:\Imperator)
├── main.py              # Punto de entrada: QApplication + MainWindow
├── requirements.txt
├── README.md
├── PLAN.md
├── .gitignore
├── ui/                  # Capa de presentación (PyQt6)
│   ├── __init__.py
│   ├── main_window.py   # Ventana principal
│   └── ...              # Ventanas/diálogos de motores, APIs y tareas
├── core/                # Núcleo de la plataforma
│   ├── __init__.py
│   ├── config_manager.py  # Lectura/escritura de JSON con validación
│   ├── engine_manager.py  # Registro y ciclo de vida de motores locales
│   ├── task_manager.py    # Descubrimiento y registro de tareas (plugins)
│   ├── api_manager.py     # Registro y ciclo de vida de APIs externas
│   └── ...
├── engines/             # Motores locales (uno por fichero)
│   ├── __init__.py
│   ├── base_engine.py   # Interfaz común Engine
│   ├── ollama.py
│   ├── llama_cpp.py
│   ├── lmstudio.py
│   └── ...              # exllamav2, vllm, tensorrt_llm
├── apis/                # APIs externas (uno por fichero)
│   ├── __init__.py
│   ├── base_api.py      # Interfaz común ApiClient
│   ├── openai.py
│   ├── deepseek.py
│   └── ...              # kling, etc.
├── tasks/               # Tareas tipo plugin (uno por fichero)
│   ├── __init__.py
│   ├── base_task.py     # Interfaz común Task
│   ├── chat.py
│   ├── image.py
│   ├── video.py
│   ├── audio.py
│   └── docs.py
├── config/              # Configuración real (ignorada por git salvo *.example.json)
│   ├── engines.example.json
│   ├── apis.example.json
│   └── settings.example.json
└── logs/                # Logs de ejecución (ignorada por git)
```

## Fases del desarrollo

- **Fase 0 — Documento PLAN.md** (esta sesión): definir objetivo, arquitectura, estructura, fases y convenciones.
- **Fase 1 — Skeleton mínimo** (esta sesión): `main.py` + ventana que abre (`ui/main_window.py`), `requirements.txt`, `README.md`, `.gitignore`.
- **Fase 2 — `config_manager.py`**: lectura/escritura de JSON con validación (esquema de motores, APIs y ajustes).
- **Fase 3 — Vertical slice**: chat con Ollama end-to-end (una tarea de chat + UI mínima que habla con el motor local).
- **Fase 4 — `engine_manager.py` + UI de motores**: registrar, listar y gestionar motores locales desde la interfaz.
- **Fase 5 — `task_manager.py` + refactor del chat a plugin**: el chat pasa a ser una tarea registrada dinámicamente.
- **Fase 6 — Añadir tareas**: image, video, audio, docs (una por sesión).
- **Fase 7 — `api_manager.py` + UI de APIs**: registro y gestión de APIs externas (OpenAI, DeepSeek, Kling, ...).
- **Fase 8 — Motores adicionales**: llama.cpp, LM Studio, exllamav2, vllm, TensorRT-LLM.

## Convenciones

### Nombres de archivos
- Módulos y paquetes en `snake_case`: `main_window.py`, `config_manager.py`, `base_engine.py`.
- Un motor local = un fichero en `engines/` con el nombre del motor: `ollama.py`, `llama_cpp.py`, `lmstudio.py`.
- Una API externa = un fichero en `apis/`: `openai.py`, `deepseek.py`, `kling.py`.
- Una tarea = un fichero en `tasks/`: `chat.py`, `image.py`, `video.py`, `audio.py`, `docs.py`.
- Configuración real en `config/*.json` (ignorada); plantillas en `config/*.example.json` (versionadas).

### Formato de logs
- Logging de la stdlib (`logging`), con formato único: `%(asctime)s %(levelname)s [%(name)s] %(message)s`.
- Nivel por defecto `INFO`; se puede subir a `DEBUG` vía configuración.
- Los logs se escriben en consola y, opcionalmente, en `logs/plataformaia.log` (rotación diaria).

### Cómo se registra un motor
- Heredar de `engines.base_engine.Engine` e implementar su interfaz (identificador, nombre, estado, `list_models()`, `generate()`/`chat()`).
- Registrarse en `engine_manager.py` exponiendo un identificador único (`ollama`, `llama_cpp`, ...).

### Cómo se registra una API
- Heredar de `apis.base_api.ApiClient` e implementar su interfaz (identificador, nombre, `chat()`, `generate_image()`, ...).
- Las claves se leen de `config/apis.json` o de variables de entorno (`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, ...); nunca en el código.

### Cómo se registra una tarea
- Heredar de `tasks.base_task.Task` e implementar su interfaz (identificador, nombre, `run()`).
- `task_manager.py` descubre y registra las tareas automáticamente para exponerlas en la UI.
