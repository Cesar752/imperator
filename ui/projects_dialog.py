"""Diálogo «Proyectos»: cola de trabajos de vídeo con miniatura, abrir y borrar.

Lista los proyectos de ``projects/`` (ver :mod:`services.video_project`), muestra
una **miniatura** de cada uno (primer fotograma de su secuencia o de su primer
clip) y permite **abrir** uno en la pestaña Vídeo o **borrarlo** del disco.

El borrado es explícito a propósito: ``projects/`` está fuera de la retención de
``output/`` para no perder intermedios, así que liberar disco es una decisión del
usuario y se confirma en un diálogo.
"""

import logging
from pathlib import Path

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from core.frame_extractor import FrameExtractor
from engines.base_engine import EngineError
from services import video_project as vproject
from services import video_script as vs

logger = logging.getLogger(__name__)


class ProjectsDialog(QDialog):
    """Lista de proyectos con miniatura; ``selected_project`` es la carpeta elegida."""

    def __init__(self, parent=None, root=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Proyectos")
        self.resize(520, 420)
        #: Carpeta del proyecto que el usuario decidió abrir (o ``None``).
        self.selected_project: Path | None = None
        #: Raíz de proyectos (``None`` = la del repositorio); inyectable en tests.
        self._root = root
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        self._list = QListWidget()
        self._list.setIconSize(QSize(96, 54))
        self._list.itemDoubleClicked.connect(lambda _item: self._open_selected())

        self._open_button = QPushButton("Abrir")
        self._open_button.clicked.connect(self._open_selected)
        self._delete_button = QPushButton("Borrar")
        self._delete_button.clicked.connect(self._delete_selected)
        self._close_button = QPushButton("Cerrar")
        self._close_button.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addWidget(self._open_button)
        buttons.addWidget(self._delete_button)
        buttons.addStretch(1)
        buttons.addWidget(self._close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self._list)
        layout.addLayout(buttons)

    def _refresh(self) -> None:
        self._list.clear()
        for folder in vproject.list_projects(self._root):
            try:
                project = vproject.load_project(folder)
            except vs.ScriptError as exc:
                logger.debug("Proyecto ignorado (%s): %s", folder, exc)
                continue
            title = project.script.title or folder.name
            item = QListWidgetItem(title)
            item.setData(Qt.ItemDataRole.UserRole, str(folder))
            item.setToolTip(
                f"{len(project.script.scenes)} escenas · {project.script.aspect} · {folder}"
            )
            thumbnail = self._thumbnail(project)
            if thumbnail is not None:
                item.setIcon(QIcon(thumbnail))
            self._list.addItem(item)

    @staticmethod
    def _thumbnail(project: vproject.Project) -> str | None:
        """Miniatura cacheada del proyecto (secuencia o primer clip), o ``None``."""
        sequence = project.directory / "secuencia.mp4"
        clips = project.saved_clips()
        source = sequence if sequence.is_file() else (clips[0] if clips else None)
        if source is None:
            return None
        mini = project.directory / "miniatura_proyecto.png"
        if not mini.is_file():
            try:
                FrameExtractor(project.directory).extract_frame(source, 0.1, mini)
            except EngineError as exc:
                logger.debug("Sin miniatura para %s: %s", project.directory, exc)
                return None
        return str(mini) if mini.is_file() else None

    def _current_folder(self) -> Path | None:
        item = self._list.currentItem()
        if item is None:
            return None
        data = item.data(Qt.ItemDataRole.UserRole)
        return Path(str(data)) if data else None

    def _open_selected(self) -> None:
        folder = self._current_folder()
        if folder is None:
            return
        self.selected_project = folder
        self.accept()

    def _delete_selected(self) -> None:
        folder = self._current_folder()
        if folder is None:
            return
        answer = QMessageBox.question(
            self,
            "Borrar proyecto",
            f"¿Borrar el proyecto «{folder.name}» y todos sus clips?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        vproject.delete_project(folder)
        self._refresh()
