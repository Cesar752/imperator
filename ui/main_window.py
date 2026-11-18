"""Ventana principal de Imperator."""

import logging

from PyQt6 import sip
from PyQt6.QtCore import Qt, QThread, QTimer
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
from core.frame_cache import FrameCache
from core.gpu_scheduler import GpuScheduler
from core.paths import ASSETS, OUTPUT
from core.task_manager import TaskManager
from services import chat_engine as ce
from services import intent_router as ir
from services.output_cleanup import (
    DEFAULT_KEEP_RECENT_HOURS,
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_MAX_FILES,
    DEFAULT_MAX_MB,
    cleanup_output,
)
from ui.engines_dialog import EnginesDialog
from ui.theme import apply_theme
from ui.workers import run_pool_task

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
        # Planificador de GPU: decide si un trabajo pesado puede empezar ahora
        # (prioridades + VRAM libre medida con nvidia-smi). Sustituye a ojo por
        # datos; las tareas lo consultan vía ``MainWindow.gpu_scheduler``.
        self._gpu_scheduler = GpuScheduler.from_settings(
            self._engine_manager.config_manager.load_settings()
        )
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
        self._cleanup_output()

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
        if not icon_path.exists():
            logger.warning("Falta el icono: %s", icon_path)
            return
        icon = QIcon(str(icon_path))
        # Si el .ico no trae imágenes usables, la barra de tareas de Windows lo
        # muestra en blanco (ver tools/make_icon.py).
        if icon.isNull() or not icon.availableSizes():
            logger.warning("El icono no tiene imágenes utilizables: %s", icon_path)
        self.setWindowIcon(icon)
        # Forzar el icono en el HWND nativo ANTES de mostrar la ventana (Qt 6 a
        # veces no propaga setWindowIcon y la barra de tareas usa el del exe).
        self._force_native_icon(str(icon_path))

    def _force_native_icon(self, icon_path: str) -> None:
        """Fija el icono en el HWND nativo vía WM_SETICON (Win32), antes de mostrar."""
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            user32.LoadImageW.argtypes = [
                wintypes.HINSTANCE,
                wintypes.LPCWSTR,
                wintypes.UINT,
                ctypes.c_int,
                ctypes.c_int,
                wintypes.UINT,
            ]
            user32.LoadImageW.restype = wintypes.HANDLE
            user32.GetSystemMetrics.argtypes = [ctypes.c_int]
            user32.GetSystemMetrics.restype = ctypes.c_int
            user32.SendMessageW.argtypes = [
                wintypes.HWND,
                wintypes.UINT,
                wintypes.WPARAM,
                wintypes.LPARAM,
            ]
            user32.SendMessageW.restype = wintypes.LPARAM

            IMAGE_ICON = 1
            LR_LOADFROMFILE = 0x00000010
            SM_CXICON = 11
            SM_CXSMALLICON = 49
            ICON_SMALL = 0
            ICON_BIG = 1
            WM_SETICON = 0x0080

            hwnd = int(self.winId())
            for flag, metric in ((ICON_BIG, SM_CXICON), (ICON_SMALL, SM_CXSMALLICON)):
                size = user32.GetSystemMetrics(metric)
                hicon = user32.LoadImageW(
                    None, icon_path, IMAGE_ICON, size, size, LR_LOADFROMFILE
                )
                if hicon:
                    user32.SendMessageW(hwnd, WM_SETICON, flag, hicon)
        except Exception as exc:  # noqa: BLE001 - el icono no debe romper el arranque
            logger.warning("No se pudo fijar el icono nativo: %s", exc)

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

        gpu_action = engines_menu.addAction("Estado de la GPU…")
        gpu_action.triggered.connect(self._show_gpu_status)

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
        # Se recuerda el widget y el índice de cada tarea por su id: el chat
        # necesita saber a qué pestaña desviar un mensaje sin duplicar el orden
        # del registro (TASK_FACTORY sigue siendo la única fuente de verdad).
        self._task_widgets: dict[str, QWidget] = {}
        self._task_tabs: dict[str, int] = {}
        for task in self._task_manager.list_tasks():
            widget = task.build_widget(parent=tabs)
            self._task_widgets[task.id] = widget
            self._task_tabs[task.id] = tabs.addTab(widget, task.nombre)
        self._tabs = tabs
        self._connect_chat_router()
        self._connect_video_to_assembler()

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
        engine = ce.preferred_chat_engine(self._engine_manager)
        nombre = getattr(engine, "nombre", "Chat") if engine is not None else "Chat"
        if engine is None or not engine.is_available():
            status = f"{nombre}: desconectado"
        else:
            status = f"{nombre}: conectado"
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
        engine = ce.preferred_chat_engine(self._engine_manager)
        # Duck-typing en lugar de isinstance(ChatTask): pedimos la capacidad
        # "refresh_engine" solo si la tarea de chat la expone. Así no acoplamos
        # MainWindow a ChatTask ni mantenemos una referencia con estado que
        # habría que re-sincronizar si cambiara el registro de tareas.
        chat_task = self._task_manager.get_task("chat")
        if chat_task is not None and hasattr(chat_task, "refresh_engine"):
            chat_task.refresh_engine()
        nombre = getattr(engine, "nombre", "Chat") if engine is not None else "Chat"
        if engine is None or not engine.is_available():
            self.statusBar().showMessage(f"{nombre}: desconectado")
        else:
            self.statusBar().showMessage(f"{nombre}: conectado")

    # ---- Router de intención (chat -> otras tareas) ---------------

    def _connect_chat_router(self) -> None:
        """Conecta la señal ``routed`` del chat con esta ventana.

        El chat solo sabe que un mensaje apunta a otra tarea; abrir la pestaña es
        responsabilidad de la ventana. Se usa ``getattr`` para no acoplar el
        widget de chat a este mecanismo (si no expone la señal, no pasa nada).
        """
        chat = self._task_widgets.get("chat")
        signal = getattr(chat, "routed", None)
        if signal is not None:
            signal.connect(self._on_chat_routed)

    def _on_chat_routed(self, dispatch) -> None:
        """Lleva el mensaje del chat a su pestaña.

        Si el dispatch trae lo necesario para **arrancar** la tarea (hoy: el
        guion de escenas), se abre la pestaña y se lanza; si no, se ofrece
        abrirla, como siempre.
        """
        index = self._task_tabs.get(getattr(dispatch, "task_id", ""))
        if index is None:
            return
        if self._start_routed_task(dispatch):
            self._tabs.setCurrentIndex(index)
            return
        if index == self._tabs.currentIndex():
            return
        # El router se dispara dentro del MISMO evento que el ``returnPressed``
        # de la barra del chat: si el diálogo se abriera aquí, Qt le entregaría la
        # tecla Enter que acaba de enviar el mensaje y se cerraría solo (aparecía
        # y desaparecía en un parpadeo, aceptando sin querer). Se difiere al
        # siguiente ciclo del bucle de eventos para que la tecla ya no esté viva.
        QTimer.singleShot(0, lambda: self._ask_open_tab(dispatch, index))

    def _ask_open_tab(self, dispatch, index: int) -> None:
        """Pregunta si abrir la pestaña destino del ``Dispatch`` (ver arriba)."""
        answer = QMessageBox.question(
            self,
            "Abrir otra tarea",
            f"{dispatch.suggestion()}\n\n"
            f"¿Abrir la pestaña «{self._tabs.tabText(index)}»?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._tabs.setCurrentIndex(index)

    def _start_routed_task(self, dispatch) -> bool:
        """Arranca la tarea destino si el dispatch trae lo necesario.

        Hoy solo el guion de vídeo lo trae (``Dispatch.payload``): idea + nº de
        escenas. Ningún widget conoce el router ni al revés; la ventana es la que
        empalma ambos con ``getattr``.
        """
        if getattr(dispatch, "task_id", "") != "video":
            return False
        if getattr(dispatch, "action", "") != ir.ACTION_PLAN_SCRIPT:
            return False
        payload = getattr(dispatch, "payload", None) or {}
        idea = str(payload.get("idea") or getattr(dispatch, "text", "") or "").strip()
        if not idea:
            return False
        video = self._task_widgets.get("video")
        start = getattr(video, "start_script", None)
        if not callable(start):
            return False
        start(idea, payload.get("scenes"))
        return True

    # ---- Vídeo -> Ensamblador ------------------------------------

    #: Pestañas que pueden mandar su vídeo montado al Ensamblador. Vídeo manda la
    #: secuencia de escenas; Vídeo IA, lo que sale de un replay, una rampa, unos
    #: rótulos o una cámara lenta; Crónica, el vídeo con narración y subtítulos.
    _ASSEMBLER_SENDERS = ("video", "video_enhance", "chronicle")

    def _connect_video_to_assembler(self) -> None:
        """Conecta la señal ``clips_ready`` de las pestañas que montan vídeo.

        Ni esos widgets conocen el Ensamblador ni al contrario: emiten la ruta de
        su vídeo y la ventana la precarga en el Ensamblador. Se usa ``getattr``
        para no acoplar los widgets a este mecanismo (y para tolerar que una
        pestaña no exista o no exponga la señal).
        """
        for task_id in self._ASSEMBLER_SENDERS:
            widget = self._task_widgets.get(task_id)
            signal = getattr(widget, "clips_ready", None)
            if signal is not None:
                signal.connect(self._on_video_clips_ready)

    def _on_video_clips_ready(self, path) -> None:
        """Precarga la secuencia generada en el Ensamblador y abre su pestaña."""
        assembler = self._task_widgets.get("assembler")
        set_video = getattr(assembler, "set_video", None)
        if path and callable(set_video):
            set_video(path)
        index = self._task_tabs.get("assembler")
        if index is not None:
            self._tabs.setCurrentIndex(index)

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
        # ``update_settings`` hace el read-modify-write bajo lock: antes se leía
        # y escribía a mano y dos cambios simultáneos (otro widget guardando una
        # preferencia) podían perder uno de los dos.
        self._engine_manager.config_manager.update_settings(
            {"unload_on_tab_change": checked}
        )

    def _theme_is_dark(self) -> bool:
        settings = self._engine_manager.config_manager.load_settings()
        return settings.get("theme", "dark") != "light"

    def _on_theme_toggled(self, checked: bool) -> None:
        theme = "dark" if checked else "light"
        self._engine_manager.config_manager.update_settings({"theme": theme})
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, theme)

    def _current_font_size(self) -> int:
        return int(self._engine_manager.config_manager.load_settings().get("font_size", 11))

    def _on_font_size(self, size: int) -> None:
        self._engine_manager.config_manager.update_settings({"font_size": size})
        app = QApplication.instance()
        if app is not None:
            font = app.font()
            font.setPointSize(size)
            app.setFont(font)

    def _save_geometry(self) -> None:
        try:
            g = self.normalGeometry()
            self._engine_manager.config_manager.update_settings(
                {
                    "geometry": {
                        "x": g.x(),
                        "y": g.y(),
                        "width": g.width(),
                        "height": g.height(),
                        "maximized": self.isMaximized(),
                    }
                }
            )
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

    def release_engines(self, except_ids=()) -> bool:
        """Pide a los motores que suelten su VRAM (salvo los de ``except_ids``).

        Lo usan el botón *Liberar VRAM ahora* y las tareas pesadas justo antes de
        empezar (``ui.workers.gpu_slot``): cuando la GPU está casi llena, lo que
        sobra suele ser la VRAM que retiene otro motor. Devuelve ``True`` si
        alguno liberó algo.
        """
        excluidos = set(except_ids)
        freed = False
        for engine_id in self._engine_manager.supported_types():
            if engine_id in excluidos:
                continue
            try:
                engine = self._engine_manager.get_engine(engine_id)
                if engine is not None and hasattr(engine, "unload_model"):
                    if engine.unload_model():
                        freed = True
            except Exception as exc:  # noqa: BLE001 - no bloquear la liberación
                logger.warning("No se pudo liberar VRAM de %s: %s", engine_id, exc)
        return freed

    def _free_all_vram(self) -> None:
        """Libera la VRAM de todos los motores que la usan (botón manual)."""
        freed = self.release_engines()
        self.statusBar().showMessage(
            "VRAM liberada" if freed else "No se pudo liberar VRAM"
        )

    # ---- Estado de la GPU ----------------------------------------

    @property
    def gpu_scheduler(self) -> GpuScheduler:
        """Planificador de GPU compartido (lo consultan las tareas pesadas)."""
        return self._gpu_scheduler

    def _show_gpu_status(self) -> None:
        """Muestra el estado de la GPU y los trabajos en curso.

        El sondeo (``nvidia-smi``) se lanza en el pool y el diálogo se abre desde
        ``_on_gpu_status``, que es un método de este ``QObject``: Qt entrega la
        señal en el hilo de la UI. Así no se bloquea la ventana esperando al
        subproceso.
        """
        run_pool_task(self._gpu_scheduler.describe, on_result=self._on_gpu_status)

    def _on_gpu_status(self, texto: str) -> None:
        if sip.isdeleted(self):
            return
        QMessageBox.information(self, "Estado de la GPU", texto)

    # ---- Retención de las salidas --------------------------------

    def _cleanup_output(self) -> None:
        """Aplica la política de retención de ``output/`` sin bloquear la UI.

        Cada imagen, vídeo, audio o SRT se queda ahí para siempre, así que este es
        el único tope (auditoría: 180 ficheros, 110,9 MB, sin límite). Se hace al
        arrancar, cuando no hay nada abierto en la interfaz: borrar en caliente
        podría llevarse el fichero que el usuario está viendo. Con la
        configuración por defecto (500 ficheros, 90 días, 8 GB, 24 h de gracia)
        no se borra nada en un ``output/`` normal; el trabajo va al pool porque
        recorre el directorio en disco.
        """
        ajustes = self._engine_manager.config_manager.load_settings().get(
            "output_limpieza"
        )
        if isinstance(ajustes, dict) and ajustes.get("enabled") is False:
            logger.info("Limpieza de salidas desactivada en settings.json.")
            return
        opciones = ajustes if isinstance(ajustes, dict) else {}
        run_pool_task(
            cleanup_output,
            OUTPUT,
            max_files=opciones.get("max_files", DEFAULT_MAX_FILES),
            max_age_days=opciones.get("max_age_days", DEFAULT_MAX_AGE_DAYS),
            max_mb=opciones.get("max_mb", DEFAULT_MAX_MB),
            keep_recent_hours=opciones.get(
                "keep_recent_hours", DEFAULT_KEEP_RECENT_HOURS
            ),
            on_result=self._on_cleanup_done,
        )
        # La caché de fotogramas (cache/frames) también se poda: sin tope, analizar
        # varios vídeos deja miles de PNG. Tope holgado; se borra lo más antiguo.
        run_pool_task(FrameCache().prune, on_result=self._on_cache_pruned)

    def _on_cleanup_done(self, plan) -> None:
        """Deja constancia de lo borrado (la UI ya no muestra esos ficheros)."""
        if plan.count:
            logger.info("Limpieza de salidas: %s", plan.describe())

    def _on_cache_pruned(self, borrados) -> None:
        """Registra la poda de la caché de fotogramas (0 = nada que borrar)."""
        if borrados:
            logger.info("Caché de fotogramas: %d ficheros antiguos borrados.", borrados)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "Acerca de",
            "PlataformaIA\nPlataforma modular de IA (Python + PyQt6).",
        )
