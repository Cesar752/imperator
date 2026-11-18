"""Receptores seguros para callbacks del pool (C2: obsolescencia).

``run_pool_task`` conecta el callback directamente a una señal de Qt. Cuando el
callback es un callable suelto (lambda, ``functools.partial``, cierre) y el
widget muere antes de que llegue el resultado, el callback se ejecuta igual
contra un objeto destruido: ``RuntimeError: wrapped C/C++ object ... has been
deleted`` o, en este proyecto, un ``TypeError`` sin traza que PyQt no explica.

La forma canónica de Qt es que el **receptor de la señal sea un ``QObject``**
hijo del widget: al destruirse el widget se destruye el receptor y Qt rompe la
conexión, así que el resultado tardío ni siquiera llega a invocarse. El token de
generación no viaja como argumento —se guarda en el receptor y se compara al
recibir—, con lo que no hay callables sueltos ni firmas especiales.
"""

from PyQt6 import sip
from PyQt6.QtCore import QObject


class PoolCallbackReceiver(QObject):
    """Recibe un resultado del pool y solo lo entrega si el widget sigue vivo.

    ``owner`` es el widget: se usa como padre (para que Qt destruya el receptor
    con él) y como fuente del token ``_generation``.
    """

    def __init__(self, owner, on_value, on_error, context=None) -> None:
        super().__init__(owner)
        self._owner = owner
        self._on_value = on_value
        self._on_error = on_error
        #: Dato extra para callbacks indexados por motor (p. ej. ``engine_id``).
        self._context = context
        self._token = 0

    def arm(self) -> int:
        """Fija el token actual del widget. Se llama justo al lanzar la tarea."""
        self._token = getattr(self._owner, "_generation", 0)
        return self._token

    def handle(self, value) -> None:
        """Entrega ``value`` al manejador del widget si sigue vivo y vigente."""
        if self._vivo():
            self._on_value(self._arg(), value)

    def handle_error(self, message) -> None:
        """Entrega el error al manejador del widget si sigue vivo y vigente."""
        if self._vivo():
            self._on_error(self._arg(), message)

    def _arg(self):
        """Primer argumento del manejador: el contexto o el token del widget."""
        return self._context if self._context is not None else self._token

    def _vivo(self) -> bool:
        if sip.isdeleted(self._owner):
            return False
        return self._token == getattr(self._owner, "_generation", self._token)
