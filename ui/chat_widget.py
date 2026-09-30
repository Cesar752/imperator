"""Widget de chat: entrada del usuario y respuesta del motor en markdown."""

import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from engines.base_engine import ChatEngine
from services import chat_params as cp
from services import preferences as prefs
from ui import markdown as md
from ui.task_widget import TaskWidget
from ui.workers import ThreadWorker, run_pool_task

logger = logging.getLogger(__name__)

# Colores de las etiquetas de autor. Son tonos medios, legibles tanto sobre el
# tema oscuro de la app como sobre el claro.
_COLOR_USUARIO = "#4a9eff"
_COLOR_MOTOR = "#69c07a"
_COLOR_ERROR = "#e06c75"
_COLOR_AVISO = "#d8a657"


class ChatWorker(ThreadWorker):
    """Ejecuta ``engine.chat()`` fuera del hilo de la UI."""

    def __init__(self, engine: ChatEngine, messages: list[dict], model: str | None, parent=None) -> None:
        super().__init__(parent)
        self._engine = engine
        self._messages = messages
        self._model = model

    def work(self) -> str:
        return self._engine.chat(self._messages, model=self._model)


class ChatWidget(TaskWidget):
    """Área de conversación, indicador de actividad, input y botón Enviar.

    Señales:
    - ``routed(object)``: el orquestador ha detectado que el mensaje apunta a
      otra tarea (documentos, imagen, vídeo…). El widget no abre pestañas: eso es
      cosa de la ventana principal, que conecta esta señal.
    """

    #: Emite el ``Dispatch`` de ``services.task_orchestrator`` cuando el mensaje
    #: corresponde a otra tarea. Se declara aquí (y no en la base) porque el chat
    #: es el único que enruta.
    routed = pyqtSignal(object)

    def __init__(
        self,
        engine: ChatEngine,
        config_manager=None,
        parent=None,
        orchestrator=None,
    ) -> None:
        super().__init__(parent)
        self._engine: ChatEngine | None = engine
        self._config_manager = config_manager
        self._orchestrator = orchestrator
        self._prefs = prefs.PrefsStore(config_manager)
        # Historial completo (lo que ve el usuario). Al motor solo se le envía
        # una ventana reciente: ver services.chat_params.MAX_HISTORY_MESSAGES.
        self._messages: list[dict] = []
        # Modelo que se envió en el turno en curso (para etiquetar la respuesta
        # con el modelo real, aunque el usuario cambie el combo mientras espera).
        self._pending_model: str | None = None
        self._worker: ChatWorker | None = None
        self._model_task = None
        self._build_ui()
        self._populate_models()

    def _build_ui(self) -> None:
        self._model = QComboBox()
        self._model.currentIndexChanged.connect(self._on_model_changed)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Modelo:"))
        model_row.addWidget(self._model, 1)

        # QTextEdit (no QPlainTextEdit) para poder insertar como texto
        # enriquecido el markdown que devuelve el motor.
        self._conversation = QTextEdit()
        self._conversation.setReadOnly(True)
        self._conversation.setAcceptRichText(True)
        self._conversation.setPlaceholderText("La conversación aparecerá aquí.")

        self._thinking_label = QLabel("Pensando...")
        self._thinking_label.setVisible(False)

        self._input = QLineEdit()
        self._input.setPlaceholderText("Escribe tu mensaje y pulsa Enter...")

        self._send_button = QPushButton("Enviar")

        input_row = QHBoxLayout()
        input_row.addWidget(self._input)
        input_row.addWidget(self._send_button)

        layout = QVBoxLayout(self)
        layout.addLayout(model_row)
        layout.addWidget(self._conversation)
        layout.addWidget(self._thinking_label)
        layout.addLayout(input_row)

        self._input.returnPressed.connect(self._send_message)
        self._send_button.clicked.connect(self._send_message)

    def set_engine(self, engine: ChatEngine | None) -> None:
        """Cambia el motor activo. Limpia el historial y el área."""
        self._engine = engine
        self._messages.clear()
        self._pending_model = None
        self._conversation.clear()
        self._set_busy(False)
        self._populate_models()

    # ---- Modelos ------------------------------------------------

    def _populate_models(self) -> None:
        """Lanza la carga de modelos en segundo plano y restaura el último."""
        self._model.blockSignals(True)
        self._model.clear()
        self._model.blockSignals(False)
        if self._engine is None:
            return
        self._model.setEnabled(False)
        self._model.addItem("Cargando modelos…")
        self._model_task = run_pool_task(
            self._engine.list_model_details,
            on_result=self._on_model_details,
            on_error=self._on_model_details_error,
        )

    def _on_model_details_error(self, message: str) -> None:
        logger.debug("No se pudieron listar los modelos de Ollama: %s", message)
        self._on_model_details([])

    def _on_model_details(self, details: list) -> None:
        """Rellena el selector con los modelos recibidos del worker."""
        details = [dict(detail) for detail in details]
        names = [detail["name"] for detail in details]
        default = getattr(self._engine, "default_model", None)
        if default and default not in names:
            details.append({"name": default, "size": None, "installed": False})
            logger.warning("default_model '%s' no está instalado en Ollama.", default)

        self._model.blockSignals(True)
        self._model.clear()
        for detail in details:
            name = detail["name"]
            installed = detail.get("installed", True)
            size = detail.get("size")
            if installed:
                display = name
                tooltip = f"{name} ({self._format_size(size)})" if size else name
            else:
                display = f"{name} (no instalado)"
                tooltip = f"{name} (no instalado en Ollama)"
            self._model.addItem(display, name)
            self._model.setItemData(
                self._model.count() - 1, tooltip, Qt.ItemDataRole.ToolTipRole
            )

        last = self._prefs.get_str(prefs.CHAT)
        if last is not None:
            idx = self._model.findData(last)
            if idx >= 0:
                self._model.setCurrentIndex(idx)
            else:
                logger.warning(
                    "Modelo guardado '%s' ya no existe en Ollama; "
                    "usando el primero del combo.",
                    last,
                )
        self._model.blockSignals(False)
        self._model.setEnabled(True)

    @staticmethod
    def _format_size(size: int) -> str:
        return f"{size / (1024 ** 3):.1f} GB"

    def _on_model_changed(self) -> None:
        model = self._model.currentData()
        if not model:
            return
        self._prefs.set(prefs.CHAT, model)
        if self._messages:
            answer = QMessageBox.question(
                self,
                "Cambiar de modelo",
                "¿Limpiar la conversación al cambiar de modelo?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer == QMessageBox.StandardButton.Yes:
                self._messages.clear()
                self._conversation.clear()

    def _send_message(self) -> None:
        if self._engine is None:
            self._append_turn("Error", "no hay motor activo.", _COLOR_ERROR)
            return
        text = self._input.text().strip()
        if not text:
            return
        if self._worker is not None and self._worker.isRunning():
            return

        self._append_turn("Tú", text, _COLOR_USUARIO)
        self._messages.append({"role": "user", "content": text})
        self._input.clear()

        self._thinking_label.setVisible(True)
        self._set_busy(True)

        self._pending_model = self._model.currentData()
        self._worker = ChatWorker(
            self._engine,
            cp.trim_history(self._messages),
            self._pending_model,
            self,
        )
        self._worker.result.connect(self._on_reply)
        self._worker.failed.connect(self._on_error)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()
        # El orquestador decide si el mensaje apunta a otra tarea. Se emite con
        # la petición ya lanzada: el diálogo que abra la ventana principal usa un
        # bucle de eventos propio y la respuesta sigue llegando mientras tanto.
        self._maybe_route(text)

    def _maybe_route(self, text: str) -> None:
        """Emite ``routed`` si el orquestador ve en ``text`` otra tarea."""
        if self._orchestrator is None:
            return
        dispatch = self._orchestrator.dispatch(text)
        if dispatch is None:
            return
        logger.debug(
            "Chat: mensaje desviado a '%s' (%s, confianza %.2f)",
            dispatch.task_id,
            dispatch.action or "abrir",
            dispatch.confidence,
        )
        self.routed.emit(dispatch)

    def _on_reply(self, reply: str) -> None:
        reply = (reply or "").strip()
        if not reply:
            # Un turno vacío no aporta nada y además ensuciaría el contexto del
            # siguiente mensaje (se enviaría un mensaje de asistente vacío).
            self._append_turn(
                "Aviso", "el motor devolvió una respuesta vacía.", _COLOR_AVISO
            )
            self._end_turn()
            return
        self._messages.append({"role": "assistant", "content": reply})
        self._append_turn(self._motor_label(), reply, _COLOR_MOTOR)
        self._end_turn()

    def _motor_label(self) -> str:
        """Etiqueta de la respuesta: el modelo que la generó.

        Orden de preferencia: el modelo que se envió (el combo puede haber
        cambiado mientras el motor respondía), el seleccionado ahora, el modelo
        por defecto del motor (mientras el combo aún carga) y, por último, el
        nombre del motor.
        """
        candidatos = (
            self._pending_model,
            self._model.currentData(),
            getattr(self._engine, "default_model", None),
            getattr(self._engine, "nombre", None),
        )
        for candidato in candidatos:
            if candidato:
                return str(candidato)
        return "Motor"

    def _on_cancelled(self, message: str) -> None:
        self._append_turn("Aviso", message, _COLOR_AVISO)
        self._end_turn()

    def _on_failed(self, message: str) -> None:
        self._append_turn("Error", message, _COLOR_ERROR)
        self._end_turn()

    def _end_turn(self) -> None:
        """Cierra el turno: oculta el indicador, libera la UI y devuelve el foco."""
        self._thinking_label.setVisible(False)
        self._set_busy(False)
        self._worker = None
        self._pending_model = None
        self._restore_focus()

    def _append_turn(self, autor: str, texto: str, color: str) -> None:
        """Añade un turno con su autor resaltado y separado del anterior.

        Del cuerpo solo se deja el color del tema (no se fija), así que se lee
        igual en claro y en oscuro; el color distingue únicamente a quién habla.
        El cuerpo que trae markdown (negritas, tablas, código) se renderiza; el
        texto llano se inserta tal cual, sin pasar por HTML.
        """
        cursor = self._conversation.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self._conversation.document().isEmpty():
            cursor.insertBlock()   # cierra el turno anterior
            cursor.insertBlock()   # y deja una línea en blanco entre turnos
        etiqueta = QTextCharFormat()
        etiqueta.setFontWeight(QFont.Weight.Bold)
        etiqueta.setForeground(QColor(color))
        cursor.setCharFormat(etiqueta)
        cursor.insertText(f"{autor}: ")
        cursor.setCharFormat(QTextCharFormat())   # el cuerpo, con el color del tema
        md.insert_markdown(cursor, texto, md.is_dark(self.palette()))
        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        """Baja la conversación al último mensaje.

        Insertar texto no garantiza el desplazamiento (si el usuario movió la
        rueda, el texto nuevo se añade fuera de la vista), así que se lleva el
        cursor al final y se asegura de que sea visible. En ``QTextEdit``
        ``ensureCursorVisible`` deja el final a un margen de documento de
        distancia, así que además se fuerza la barra a su máximo.
        """
        cursor = self._conversation.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._conversation.setTextCursor(cursor)
        self._conversation.ensureCursorVisible()
        bar = self._conversation.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _busy_widgets(self) -> tuple:
        return (self._model, self._input, self._send_button)

    def _focus_widget(self):
        """La barra de escritura es el control que debe recuperar el foco."""
        return self._input
