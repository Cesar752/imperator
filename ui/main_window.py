"""Ventana principal de PlataformaIA."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QMainWindow, QMessageBox

from core.engine_manager import EngineManager
from ui.chat_widget import ChatWidget
from ui.engines_dialog import EnginesDialog


class MainWindow(QMainWindow):
    """Ventana principal con chat central y menú de gestión de motores."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PlataformaIA")
        self.resize(900, 600)
        self._engine_manager = EngineManager()
        self._chat_widget: ChatWidget | None = None
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
        engine = self._engine_manager.get_engine("ollama")
        if engine is None:
            placeholder = QLabel("No hay motor Ollama registrado o habilitado.")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setCentralWidget(placeholder)
            return
        self._chat_widget = ChatWidget(engine)
        self.setCentralWidget(self._chat_widget)

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
        if self._chat_widget is not None:
            self._chat_widget.set_engine(engine)
        if engine is None:
            self.statusBar().showMessage("Ollama: desconectado")

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "Acerca de",
            "PlataformaIA\nPlataforma modular de IA (Python + PyQt6).",
        )
