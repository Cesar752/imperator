"""Registro y resolución de las APIs externas de la plataforma (Fase 7).

ConfigManager sigue siendo el responsable de leer/escribir ``config/apis.json``;
este gestor le da un consumidor real: expone las APIs registradas y resuelve sus
claves **solo desde variables de entorno** (nunca del JSON en claro).
"""

import logging

from core.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class ApiManager:
    """Carga las APIs externas de ``config/apis.json`` y resuelve sus claves."""

    def __init__(self, config_manager: ConfigManager | None = None) -> None:
        self._config = config_manager if config_manager is not None else ConfigManager()
        self._apis: dict[str, dict] = {}
        self.reload()

    def reload(self) -> None:
        """Recarga ``apis.json``, tolerante a fallos (config corrupta)."""
        try:
            entries = self._config.load_apis()
        except Exception as exc:  # noqa: BLE001 - config corrupta no debe romper el arranque
            logger.error("No se pudo cargar apis.json (%s); se ignora.", exc)
            entries = []
        self._apis = {entry["id"]: entry for entry in entries}

    # ---- Consultas ----------------------------------------------

    def list_apis(self) -> list[dict]:
        """APIs registradas (copias, para no exponer el dict interno)."""
        return [dict(api) for api in self._apis.values()]

    def get_api(self, api_id: str) -> dict | None:
        api = self._apis.get(api_id)
        return dict(api) if api is not None else None

    def is_enabled(self, api_id: str) -> bool:
        api = self._apis.get(api_id)
        return bool(api and api.get("enabled", False))

    def supported_types(self) -> list[str]:
        """Tipos de API configurados (``llm``, ``image``, ``video``, ``audio``)."""
        return sorted(
            {api["tipo"] for api in self._apis.values() if api.get("tipo")}
        )

    def resolve_api_key(self, api_id: str) -> str:
        """Clave de la API, leída del entorno (nunca del JSON)."""
        api = self._apis.get(api_id)
        if api is None:
            return ""
        return self._config.resolve_api_key(api)
