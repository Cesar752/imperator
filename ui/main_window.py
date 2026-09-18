"""Ventana principal de PlataformaIA."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QMainWindow, QMessageBox


class MainWindow(QMainWindow):
    """Ventana vacía con menú básico y placeholder central."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PlataformaIA")
        self.resize(900, 600)
        self._build_menu()
        self._build_central_widget()

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&Archivo")
        exit_action = file_menu.addAction("&Salir")
        exit_action.triggered.connect(self.close)

        help_menu = menubar.addMenu("&Ayuda")
        about_action = help_menu.addAction("&Acerca de")
        about_action.triggered.connect(self._show_about)

    def _build_central_widget(self) -> None:
        placeholder = QLabel("PlataformaIA\n\nCarga aquí tus motores, APIs y tareas.")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(placeholder)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "Acerca de",
            "PlataformaIA\nPlataforma modular de IA (Python + PyQt6).",
        )
