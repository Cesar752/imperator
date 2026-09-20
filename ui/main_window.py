"""Ventana principal de PlataformaIA."""

import logging
from pathlib import Path

from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.engine_manager import EngineManager
from core.task_manager import TaskManager
from ui.engines_dialog import EnginesDialog

logger = logging.getLogger(__name__)

# Raíz del proyecto (C:\Imperator); los assets viven en assets/.
ASSETS = Path(__file__).resolve().parent.parent / "assets"

# Altura del logo en el header (px).
HEADER_LOGO_HEIGHT = 60

# Índice de pestaña -> id de motor Ollama cuyo modelo se descarga al salir.
# El orden de pestañas viene de TASK_FACTORY (0 chat, 1 document, ...).
TAB_TO_ENGINE = {0: "ollama", 1: "ollama_vision"}


class MainWindow(QMainWindow):
    """Ventana principal con las tareas registradas en pestañas y menú de motores."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PlataformaIA")
        self._apply_window_icon()
        self.resize(900, 600)
        self._engine_manager = EngineManager()
        self._task_manager = TaskManager(self._engine_manager)
        self._engines_dialog: EnginesDialog | None = None
        self._current_tab_index = 0
        self._unload_toggle = None
        self._build_menu()
        self._build_central_widget()
        self._validate_tab_mapping()
        self._build_status_bar()

    def closeEvent(self, event) -> None:
        # Al cerrar la ventana, detener los workers antes de que Qt destruya
        # los QThread mientras aún se ejecutan ("QThread: Destroyed while
        # thread is still running").

        # 1. Cancelación remota en motores que la soportan (Video, Image).
        for engine_id in ("wanvideo", "comfyui"):
            try:
                engine = self._engine_manager.get_engine(engine_id)
                if engine is not None and hasattr(engine, "cancel_remote"):
                    engine.cancel_remote()
            except Exception as exc:  # noqa: BLE001 - no bloquear el cierre
                logger.warning("No se pudo cancelar remoto en %s: %s", engine_id, exc)

        # 2. Interrupción de los QThread hijos.
        running = [t for t in self.findChildren(QThread) if t.isRunning()]
        for thread in running:
            thread.requestInterruption()

        # 3. Espera híbrida: 5 s con límite, luego sin límite para evitar crash.
        for thread in running:
            if not thread.wait(5000):
                logger.warning(
                    "Worker no terminó en 5s, esperando sin límite: %s", thread
                )
                thread.wait()

        event.accept()

    def _apply_window_icon(self) -> None:
        icon_path = ASSETS / "icono.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        else:
            logger.warning("Falta el icono: %s", icon_path)

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&Archivo")
        exit_action = file_menu.addAction("&Salir")
        exit_action.triggered.connect(self.close)

        engines_menu = menubar.addMenu("&Motores")
        manage_action = engines_menu.addAction("&Gestionar motores...")
        manage_action.triggered.connect(self._open_engines_dialog)

        engines_menu.addSeparator()

        self._unload_toggle = engines_menu.addAction("Liberar VRAM al cambiar de pestaña")
        self._unload_toggle.setCheckable(True)
        self._unload_toggle.setChecked(self._unload_on_tab_change_enabled())
        self._unload_toggle.setToolTip(
            "Libera la VRAM del modelo de Ollama al salir de Chat o Documentos.\n"
            "Desactivar acelera si alternas Chat/Documentos, pero puede provocar OOM\n"
            "al pasar a Imagen o Vídeo."
        )
        self._unload_toggle.toggled.connect(self._on_unload_toggle)

        free_action = engines_menu.addAction("Liberar VRAM ahora")
        free_action.triggered.connect(self._free_all_vram)

        help_menu = menubar.addMenu("&Ayuda")
        about_action = help_menu.addAction("&Acerca de")
        about_action.triggered.connect(self._show_about)

    def _build_central_widget(self) -> None:
        tabs = QTabWidget()
        tabs.currentChanged.connect(self._on_tab_changed)
        for task in self._task_manager.list_tasks():
            tabs.addTab(task.build_widget(parent=tabs), task.nombre)

        header = self._build_header()
        if header is None:
            self.setCentralWidget(tabs)
            return

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(tabs)
        self.setCentralWidget(central)

    def _build_header(self) -> QLabel | None:
        """Devuelve el QLabel con el logo, o ``None`` si no hay logo disponible."""
        logo_path = ASSETS / "imperator.png"
        if not logo_path.exists():
            logger.warning("Falta el logo: %s", logo_path)
            return None
        pixmap = QPixmap(str(logo_path))
        if pixmap.isNull():
            logger.warning("El logo no se pudo cargar: %s", logo_path)
            return None

        label = QLabel()
        label.setPixmap(
            pixmap.scaledToHeight(
                HEADER_LOGO_HEIGHT,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        label.setFixedHeight(HEADER_LOGO_HEIGHT)
        label.setContentsMargins(8, 0, 8, 0)
        return label

    def _build_status_bar(self) -> None:
        engine = self._engine_manager.get_engine("ollama")
        if engine is None or not engine.is_available():
            status = "Ollama: desconectado"
        else:
            status = "Ollama: conectado"
        self.statusBar().showMessage(status)

    def _open_engines_dialog(self) -> None:
        if self._engines_dialog is not None and self._engines_dialog.isVisible():
            self._engines_dialog.raise_()
            self._engines_dialog.activateWindow()
            return
        self._engines_dialog = EnginesDialog(self._engine_manager, self)
        self._engines_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._engines_dialog.finished.connect(self._on_engines_dialog_finished)
        self._engines_dialog.show()

    def _on_engines_dialog_finished(self) -> None:
        self._engines_dialog = None
        engine = self._engine_manager.get_engine("ollama")
        # Duck-typing en lugar de isinstance(ChatTask): pedimos la capacidad
        # "refresh_engine" solo si la tarea de chat la expone. Así no acoplamos
        # MainWindow a ChatTask ni mantenemos una referencia con estado que
        # habría que re-sincronizar si cambiara el registro de tareas.
        chat_task = self._task_manager.get_task("chat")
        if chat_task is not None and hasattr(chat_task, "refresh_engine"):
            chat_task.refresh_engine()
        if engine is None or not engine.is_available():
            self.statusBar().showMessage("Ollama: desconectado")
        else:
            self.statusBar().showMessage("Ollama: conectado")

    # ---- Liberación de VRAM --------------------------------------

    def _on_tab_changed(self, index: int) -> None:
        """Descarga el motor Ollama de la pestaña que se abandona."""
        previous = self._current_tab_index
        self._current_tab_index = index
        if previous == index or not self._unload_toggle.isChecked():
            return
        engine_id = TAB_TO_ENGINE.get(previous)
        if engine_id is None:
            return
        try:
            engine = self._engine_manager.get_engine(engine_id)
            if engine is not None and hasattr(engine, "unload_model"):
                engine.unload_model()
        except Exception as exc:  # noqa: BLE001 - no bloquear el cambio de pestaña
            logger.warning("No se pudo liberar VRAM del motor %s: %s", engine_id, exc)

    def _validate_tab_mapping(self) -> None:
        """Verifica que TAB_TO_ENGINE siga alineado con el orden de pestañas."""
        tasks = self._task_manager.list_tasks()
        expected = {0: "chat", 1: "document"}
        for tab_index, task_id in expected.items():
            if tab_index >= len(tasks) or tasks[tab_index].id != task_id:
                logger.warning(
                    "TAB_TO_ENGINE desincronizado: se esperaba '%s' en la pestaña %d.",
                    task_id,
                    tab_index,
                )

    def _unload_on_tab_change_enabled(self) -> bool:
        settings = self._engine_manager.config_manager.load_settings()
        return bool(settings.get("unload_on_tab_change", True))

    def _on_unload_toggle(self, checked: bool) -> None:
        settings = self._engine_manager.config_manager.load_settings()
        settings["unload_on_tab_change"] = checked
        self._engine_manager.config_manager.save_settings(settings)

    def _free_all_vram(self) -> None:
        """Libera la VRAM de todos los motores que la usan (botón manual)."""
        freed = False
        for engine_id in self._engine_manager.supported_types():
            try:
                engine = self._engine_manager.get_engine(engine_id)
                if engine is not None and hasattr(engine, "unload_model"):
                    if engine.unload_model():
                        freed = True
            except Exception as exc:  # noqa: BLE001 - no bloquear la liberación
                logger.warning("No se pudo liberar VRAM de %s: %s", engine_id, exc)
        self.statusBar().showMessage(
            "VRAM liberada" if freed else "No se pudo liberar VRAM"
        )

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "Acerca de",
            "PlataformaIA\nPlataforma modular de IA (Python + PyQt6).",
        )
