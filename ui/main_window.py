"""Ventana principal de Imperator."""

import logging

from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QActionGroup, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.api_manager import ApiManager
from core.engine_manager import EngineManager
from core.paths import ASSETS
from core.task_manager import TaskManager
from ui.engines_dialog import EnginesDialog
from ui.theme import apply_theme

logger = logging.getLogger(__name__)

# Altura del logo en el header (px).
HEADER_LOGO_HEIGHT = 120


class MainWindow(QMainWindow):
    """Ventana principal con las tareas registradas en pestañas y menú de motores."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Imperator")
        self._apply_window_icon()
        self.resize(1280, 720)
        self._engine_manager = EngineManager()
        self._api_manager = ApiManager(self._engine_manager.config_manager)
        logger.info(
            "APIs externas configuradas: %d", len(self._api_manager.list_apis())
        )
        # Se comparte el ApiManager ya construido: el TaskManager no vuelve a
        # leer config/apis.json (y una tarea que lo necesite lo recibirá vía
        # TaskDeps).
        self._task_manager = TaskManager(self._engine_manager, self._api_manager)
        # Índice de pestaña -> motor cuyo modelo se descarga al abandonarla. Se
        # deriva del registro de tareas (Task.engine_id) para no duplicar aquí
        # el orden de las pestañas (única fuente de verdad: TASK_FACTORY).
        self._tab_to_engine: dict[int, str] = {
            index: task.engine_id
            for index, task in enumerate(self._task_manager.list_tasks())
            if task.engine_id
        }
        self._engines_dialog: EnginesDialog | None = None
        self._current_tab_index = 0
        self._unload_toggle = None
        self._build_menu()
        self._build_central_widget()
        self._build_status_bar()
        self._restore_geometry()

    def closeEvent(self, event) -> None:
        self._save_geometry()
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

        # 3. Espera acotada: 5 s por hilo. Esperar sin límite podía colgar el
        # cierre indefinidamente si un worker quedaba atascado en red/ffmpeg.
        for thread in running:
            if not thread.wait(5000):
                logger.warning(
                    "Worker no terminó en 5s; se continúa con el cierre: %s", thread
                )

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

        view_menu = menubar.addMenu("&Ver")
        self._dark_theme_toggle = view_menu.addAction("Modo oscuro")
        self._dark_theme_toggle.setCheckable(True)
        self._dark_theme_toggle.setChecked(self._theme_is_dark())
        self._dark_theme_toggle.toggled.connect(self._on_theme_toggled)

        self._font_group = QActionGroup(self)
        self._font_group.setExclusive(True)
        font_menu = view_menu.addMenu("Tamaño de texto")
        current_size = self._current_font_size()
        for size in (10, 11, 12, 13, 14):
            action = font_menu.addAction(f"{size} pt")
            action.setCheckable(True)
            action.setActionGroup(self._font_group)
            action.setChecked(size == current_size)
            action.triggered.connect(lambda _checked, s=size: self._on_font_size(s))

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
        """Descarga el motor asociado a la pestaña que se abandona."""
        previous = self._current_tab_index
        self._current_tab_index = index
        if previous == index or not self._unload_toggle.isChecked():
            return
        engine_id = self._tab_to_engine.get(previous)
        if engine_id is None:
            return
        try:
            engine = self._engine_manager.get_engine(engine_id)
            if engine is not None and hasattr(engine, "unload_model"):
                engine.unload_model()
        except Exception as exc:  # noqa: BLE001 - no bloquear el cambio de pestaña
            logger.warning("No se pudo liberar VRAM del motor %s: %s", engine_id, exc)

    def _unload_on_tab_change_enabled(self) -> bool:
        settings = self._engine_manager.config_manager.load_settings()
        return bool(settings.get("unload_on_tab_change", True))

    def _on_unload_toggle(self, checked: bool) -> None:
        settings = self._engine_manager.config_manager.load_settings()
        settings["unload_on_tab_change"] = checked
        self._engine_manager.config_manager.save_settings(settings)

    def _theme_is_dark(self) -> bool:
        settings = self._engine_manager.config_manager.load_settings()
        return settings.get("theme", "dark") != "light"

    def _on_theme_toggled(self, checked: bool) -> None:
        theme = "dark" if checked else "light"
        settings = self._engine_manager.config_manager.load_settings()
        settings["theme"] = theme
        self._engine_manager.config_manager.save_settings(settings)
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, theme)

    def _current_font_size(self) -> int:
        return int(self._engine_manager.config_manager.load_settings().get("font_size", 11))

    def _on_font_size(self, size: int) -> None:
        settings = self._engine_manager.config_manager.load_settings()
        settings["font_size"] = size
        self._engine_manager.config_manager.save_settings(settings)
        app = QApplication.instance()
        if app is not None:
            font = app.font()
            font.setPointSize(size)
            app.setFont(font)

    def _save_geometry(self) -> None:
        try:
            g = self.normalGeometry()
            settings = self._engine_manager.config_manager.load_settings()
            settings["geometry"] = {
                "x": g.x(),
                "y": g.y(),
                "width": g.width(),
                "height": g.height(),
                "maximized": self.isMaximized(),
            }
            self._engine_manager.config_manager.save_settings(settings)
        except Exception as exc:  # noqa: BLE001 - no bloquear el cierre
            logger.warning("No se pudo guardar la geometría de la ventana: %s", exc)

    def _restore_geometry(self) -> None:
        try:
            geom = self._engine_manager.config_manager.load_settings().get("geometry")
            if not isinstance(geom, dict):
                return
            x, y = geom.get("x"), geom.get("y")
            width, height = geom.get("width"), geom.get("height")
            if all(isinstance(v, int) for v in (x, y, width, height)):
                self.setGeometry(x, y, width, height)
            if geom.get("maximized"):
                self.showMaximized()
        except Exception as exc:  # noqa: BLE001 - geometría corrupta no debe romper el arranque
            logger.warning("No se pudo restaurar la geometría de la ventana: %s", exc)

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
