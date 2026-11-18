"""Motores que envuelven **programas externos de línea de comandos**.

Es el punto de extensión para integrar herramientas que no son bibliotecas
Python ni exponen API: se les pasa la entrada por una plantilla de argumentos
(``{prompt}``, ``{input}``, ``{output}``, ``{output_dir}``, ``{model}``) y se
recoge el artefacto que producen (ver :class:`core.cli_runner.CliRunner`).

Un motor así declarado aparece en las mismas pestañas que los motores nativos
(la pestaña Vídeo ofrece los :class:`~engines.base_engine.VideoEngine`), de modo
que integrar una CLI no exige tocar la UI. Si el ejecutable no está instalado,
``is_available()`` devuelve ``False`` y el motor se muestra como no disponible,
igual que YOLO/CLIP sin sus dependencias.
"""

from __future__ import annotations

from pathlib import Path

from core.cli_runner import CliRunner
from engines.base_engine import VideoEngine


class CliVideoEngine(VideoEngine):
    """Motor de vídeo respaldado por un ejecutable externo.

    No fija ``id``/``nombre`` (los recibe de su configuración) para poder
    reutilizarlo con varias CLIs: ``bkingfilm``, ``lapian_notes`` u otras.
    """

    def __init__(
        self,
        *,
        engine_id: str,
        nombre: str,
        executable: str | None = None,
        args: str | list[str] | tuple[str, ...] | None = None,
        output_dir: str | Path | None = None,
        output_ext: str | None = None,
        output_glob: str | None = None,
        timeout: float | None = None,
        models: list[str] | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        self._id = engine_id
        self._nombre = nombre
        self._models = list(models or [])
        self._runner = CliRunner(
            executable,
            label=nombre,
            args=args,
            output_dir=output_dir,
            output_ext=output_ext,
            output_glob=output_glob,
            timeout=timeout,
            cwd=cwd,
        )

    # ---- Contrato Engine -------------------------------------------

    @property
    def id(self) -> str:
        return self._id

    @property
    def nombre(self) -> str:
        return self._nombre

    def is_available(self) -> bool:
        """``True`` si el ejecutable externo está instalado y accesible."""
        return self._runner.is_available()

    def list_models(self) -> list[str]:
        return list(self._models)

    # ---- Contrato VideoEngine --------------------------------------

    def generate_video(self, prompt: str, **kwargs) -> Path:
        """Invoca la CLI externa y devuelve el artefacto que produce.

        ``kwargs`` admite ``input_path`` (o ``image_path``) para herramientas que
        **procesan** un vídeo/imagen de entrada en vez de generarlo desde cero,
        ``model`` para elegir el modelo, ``output_dir``/``output_path`` para
        fijar el destino y ``timeout`` para acotar la ejecución.
        """
        entrada = kwargs.get("input_path") or kwargs.get("image_path")
        values = {
            "prompt": prompt,
            "input": str(entrada) if entrada else "",
            "model": kwargs.get("model") or (self._models[0] if self._models else ""),
            "output_dir": kwargs.get("output_dir") or "",
            "output": kwargs.get("output_path") or "",
        }
        return self._runner.invoke(values, timeout=kwargs.get("timeout"))
