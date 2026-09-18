"""Ventana principal de PlataformaIA."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMainWindow, QMessageBox, QTabWidget

from core.engine_manager import EngineManager
from core.task_manager import TaskManager
from ui.engines_dialog import EnginesDialog


class MainWindow(QMainWindow):
    """Ventana principal con las tareas registradas en pestañas y menú de motores."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PlataformaIA")
        self.resize(900, 600)
        self._engine_manager = EngineManager()
        self._task_manager = TaskManager(self._engine_manager)
        self._engines_dialog: EnginesDialog | None = None
        self._build_menu()
        self._build_central_widget()
        self._build_status_bar()

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&Archivo")
        exit_action = file_menu.addAction("&Salir")
        exit_action.triggered.connect(self.close)

        engines_menu = menubar.addMenu("&Motores")
        manage_action = engines_menu.addAction("&Gestionar motores...")
        manage_action.triggered.connect(self._open_engines_dialog)

        help_menu = menubar.addMenu("&Ayuda")
        about_action = help_menu.addAction("&Acerca de")
        about_action.triggered.connect(self._show_about)

    def _build_central_widget(self) -> None:
        tabs = QTabWidget()
        for task in self._task_manager.list_tasks():
            tabs.addTab(task.build_widget(parent=tabs), task.nombre)
        self.setCentralWidget(tabs)

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

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "Acerca de",
            "PlataformaIA\nPlataforma modular de IA (Python + PyQt6).",
        )
