from __future__ import annotations

from pathlib import Path
import subprocess
import time

from PyQt6.QtCore import QDir, QModelIndex, QPoint, QSortFilterProxyModel, QThreadPool, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFileSystemModel
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDockWidget,
    QFileIconProvider,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QSplitter,
    QStatusBar,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from core.cache_manager import CacheManager
from core.image_engine import ImageEngine
from core.workers import MetadataWorker, ThumbnailJob, ThumbnailWorker
from ui.thumbnail_delegate import ThumbnailDelegate


IMAGE_FILTERS = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.webp",
    "*.psd",
    "*.ims",
    "*.stk",
    "*.czi",
    "*.nd2",
    "*.tif",
    "*.tiff",
    "*.ome.tif",
    "*.ome.tiff",
]

FILE_TYPE_GROUPS = {
    "TIFF": {".tif", ".tiff", ".ome.tif", ".ome.tiff"},
    "IMS": {".ims"},
    "STK": {".stk"},
    "PSD": {".psd"},
    "PNG": {".png"},
    "JPG": {".jpg", ".jpeg"},
}


class QuickLookDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setModal(False)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(self.label)
        self.resize(900, 900)

    def set_pixmap(self, pixmap) -> None:
        self.label.setPixmap(pixmap)


class ThumbnailListView(QListView):
    viewportChanged = pyqtSignal()
    quickLookRequested = pyqtSignal(str)
    scrubRequested = pyqtSignal(str, float)
    contextMenuRequested = pyqtSignal(str, QPoint)

    @staticmethod
    def _model_file_path(model, index: QModelIndex) -> str:
        if model is None or not index.isValid():
            return ""

        if isinstance(model, QFileSystemModel):
            return model.filePath(index)

        if isinstance(model, QSortFilterProxyModel):
            source_index = model.mapToSource(index)
            return ThumbnailListView._model_file_path(model.sourceModel(), source_index)

        return ""

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self.viewportChanged.emit()

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        self.viewportChanged.emit()

    def keyPressEvent(self, e) -> None:
        if e is None:
            return

        if e.key() == Qt.Key.Key_Space:
            idx = self.currentIndex()
            if idx.isValid():
                file_path = self._model_file_path(self.model(), idx)
                if file_path:
                    self.quickLookRequested.emit(file_path)
            e.accept()
            return

        super().keyPressEvent(e)

    def mouseMoveEvent(self, e) -> None:
        super().mouseMoveEvent(e)
        if e is None:
            return

        idx = self.indexAt(e.position().toPoint())
        if not idx.isValid():
            return

        file_path = self._model_file_path(self.model(), idx)
        if not file_path:
            return

        rect = self.visualRect(idx)
        if rect.width() <= 0:
            return

        fraction = (e.position().x() - rect.left()) / rect.width()
        fraction = min(1.0, max(0.0, float(fraction)))
        self.scrubRequested.emit(file_path, fraction)

    def contextMenuEvent(self, a0) -> None:
        if a0 is None:
            return

        idx = self.indexAt(a0.pos())
        if not idx.isValid():
            return

        file_path = self._model_file_path(self.model(), idx)
        if not file_path:
            return

        self.setCurrentIndex(idx)
        self.contextMenuRequested.emit(file_path, a0.globalPos())


class ThumbnailFilterProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._search_text = ""
        self._enabled_extensions: set[str] = set()
        self._toggle_extensions: set[str] = set()

    def set_search_text(self, text: str) -> None:
        self._search_text = (text or "").strip().lower()
        self.invalidateFilter()

    def set_enabled_extensions(self, extensions: set[str]) -> None:
        self._enabled_extensions = {ext.lower() for ext in extensions}
        self.invalidateFilter()

    def set_toggle_extensions(self, extensions: set[str]) -> None:
        self._toggle_extensions = {ext.lower() for ext in extensions}
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        source_model = self.sourceModel()
        if not isinstance(source_model, QFileSystemModel):
            return True

        index = source_model.index(source_row, 0, source_parent)
        if not index.isValid():
            return False

        if source_model.isDir(index):
            return True

        file_name = source_model.fileName(index).lower()
        file_path = source_model.filePath(index).lower()

        matched_toggle_exts = [ext for ext in self._toggle_extensions if file_path.endswith(ext)]
        if matched_toggle_exts and not any(ext in self._enabled_extensions for ext in matched_toggle_exts):
            return False

        if self._search_text and self._search_text not in file_name:
            return False

        return True


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Microscopy Thumbnail Viewer")
        self.resize(1500, 920)

        self.cache_manager = CacheManager()
        self.image_engine = ImageEngine()
        self.thread_pool = QThreadPool(self)
        self.thread_pool.setMaxThreadCount(4)

        self._thumbnail_size = 160
        self._quicklook_size = 900
        self._initial_prefetch_count = 48
        self._folder_open_started_at: float = 0.0
        self._active_extensions = {ext for exts in FILE_TYPE_GROUPS.values() for ext in exts}

        self._in_flight: set[str] = set()
        self._metadata_in_flight: set[str] = set()
        self._file_metadata: dict[str, dict] = {}
        self._last_scrub: dict[str, tuple[str, int]] = {}
        self._quicklook_dialog = QuickLookDialog(self)

        self.model = QFileSystemModel(self)
        self.model.setIconProvider(QFileIconProvider())
        self.model.setRootPath("")
        self.model.setFilter(QDir.Filter.AllDirs | QDir.Filter.NoDotAndDotDot | QDir.Filter.Files)
        self.model.setNameFilters(IMAGE_FILTERS)
        self.model.setNameFilterDisables(False)

        self.list_proxy = ThumbnailFilterProxyModel(self)
        self.list_proxy.setSourceModel(self.model)
        self.list_proxy.set_enabled_extensions(self._active_extensions)
        self.list_proxy.set_toggle_extensions({ext for exts in FILE_TYPE_GROUPS.values() for ext in exts})

        self.tree_view = self._build_tree_view()
        self.list_view = self._build_list_view()
        self.thumbnail_panel = self._build_thumbnail_panel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.tree_view)
        splitter.addWidget(self.thumbnail_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        splitter.setSizes([320, 1080])

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self.setCentralWidget(container)

        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage(f"Ready — folder cache database: {self.cache_manager.cache_db_path()}")

        self._setup_metadata_dock()

        self.tree_view.setRootIndex(QModelIndex())
        self.list_view.setRootIndex(QModelIndex())

        tree_selection = self.tree_view.selectionModel()
        if tree_selection is not None:
            tree_selection.currentChanged.connect(self._on_tree_current_changed)

        list_selection = self.list_view.selectionModel()
        if list_selection is not None:
            list_selection.currentChanged.connect(self._on_list_current_changed)

        self.list_view.viewportChanged.connect(self._queue_visible_thumbnails)
        self.list_view.quickLookRequested.connect(self._open_quicklook)
        self.list_view.scrubRequested.connect(self._on_scrub_request)
        self.list_view.contextMenuRequested.connect(self._show_context_menu)
        self.list_view.setMouseTracking(True)

        self.model.directoryLoaded.connect(self._on_directory_loaded)

        self._prime_folder_thumbnails()
        self._queue_visible_thumbnails()

    def _setup_metadata_dock(self) -> None:
        self.metadata_text = QPlainTextEdit(self)
        self.metadata_text.setReadOnly(True)

        self.metadata_dock = QDockWidget("Metadata", self)
        self.metadata_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.metadata_dock.setWidget(self.metadata_text)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.metadata_dock)

    def _build_tree_view(self) -> QTreeView:
        tree = QTreeView(self)
        tree.setModel(self.model)
        tree.setUniformRowHeights(True)
        tree.setSortingEnabled(True)
        tree.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        tree.setAnimated(False)
        tree.setIndentation(16)
        tree.setColumnWidth(0, 280)
        tree.hideColumn(1)
        tree.hideColumn(2)
        tree.hideColumn(3)
        return tree

    def _build_list_view(self) -> ThumbnailListView:
        view = ThumbnailListView(self)
        view.setModel(self.list_proxy)
        view.setViewMode(QListView.ViewMode.IconMode)
        view.setLayoutMode(QListView.LayoutMode.Batched)
        view.setBatchSize(120)
        view.setFlow(QListView.Flow.LeftToRight)
        view.setResizeMode(QListView.ResizeMode.Adjust)
        view.setWrapping(True)
        view.setUniformItemSizes(True)
        view.setSpacing(10)
        view.setWordWrap(True)
        view.setSelectionMode(QListView.SelectionMode.SingleSelection)
        view.setVerticalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        delegate = ThumbnailDelegate(view)
        delegate.set_thumbnail_provider(self._get_thumbnail)
        delegate.set_thumbnail_requester(self._request_thumbnail)
        delegate.set_metadata_provider(self._get_metadata)
        view.setItemDelegate(delegate)
        return view

    def _build_thumbnail_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        controls = QWidget(panel)
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        self.search_input = QLineEdit(controls)
        self.search_input.setPlaceholderText("Search thumbnails by filename...")
        self.search_input.textChanged.connect(self._on_search_text_changed)
        controls_layout.addWidget(self.search_input, 1)

        self.type_checkboxes: dict[str, QCheckBox] = {}
        for label in ("TIFF", "IMS", "STK", "PSD", "PNG", "JPG"):
            checkbox = QCheckBox(label, controls)
            checkbox.setChecked(True)
            checkbox.toggled.connect(lambda checked, name=label: self._on_type_toggled(name, checked))
            controls_layout.addWidget(checkbox)
            self.type_checkboxes[label] = checkbox

        layout.addWidget(controls)
        layout.addWidget(self.list_view, 1)
        return panel

    def _on_search_text_changed(self, text: str) -> None:
        self.list_proxy.set_search_text(text)
        self._queue_visible_thumbnails()

    def _on_type_toggled(self, label: str, checked: bool) -> None:
        extensions = FILE_TYPE_GROUPS.get(label, set())
        if checked:
            self._active_extensions.update(extensions)
        else:
            for ext in extensions:
                self._active_extensions.discard(ext)

        self.list_proxy.set_enabled_extensions(self._active_extensions)
        self._queue_visible_thumbnails()

    def _cache_key(self, file_path: str, size: int | None = None, slice_request: dict | None = None) -> str:
        base_size = size if size is not None else self._thumbnail_size
        if slice_request:
            mode = slice_request.get("mode", "")
            idx = slice_request.get("index", 0)
            return f"thumb::{base_size}::{mode}:{idx}::{file_path}"
        return f"thumb::{base_size}::{file_path}"

    def _get_thumbnail(self, file_path: str):
        return self.cache_manager.get(self._cache_key(file_path))

    def _get_metadata(self, file_path: str) -> dict:
        return self._file_metadata.get(file_path, {})

    def _on_tree_current_changed(self, current: QModelIndex, previous: QModelIndex) -> None:
        if not current.isValid():
            return

        path = self.model.filePath(current)
        if not Path(path).is_dir():
            return

        self._folder_open_started_at = time.perf_counter()
        previous_record = self.cache_manager.get_folder_record(path)
        if previous_record:
            prev_count = previous_record.get("item_count", "?")
            prev_time = previous_record.get("last_loaded", "")
            self._status_bar.showMessage(f"Opening folder: {path} (seen before, last items: {prev_count}, at {prev_time})")
        else:
            self._status_bar.showMessage(f"Opening folder: {path} (first load)")

        source_root = self.model.index(path)
        proxy_root = self.list_proxy.mapFromSource(source_root)
        self.list_view.setRootIndex(proxy_root)

        self._prime_folder_thumbnails()
        self._queue_visible_thumbnails()

    def _on_list_current_changed(self, current: QModelIndex, previous: QModelIndex) -> None:
        if not current.isValid():
            return

        file_path = self._file_path_from_list_index(current)
        if not file_path:
            return

        if Path(file_path).is_dir():
            self.metadata_text.setPlainText(f"Path: {file_path}\n\nType: Folder")
            return

        cached = self._file_metadata.get(file_path)
        if cached:
            self._render_metadata(file_path, cached)

        if file_path in self._metadata_in_flight:
            return

        self._metadata_in_flight.add(file_path)
        worker = MetadataWorker(self.image_engine, file_path)
        worker.signals.finished.connect(self._on_metadata_ready)
        worker.signals.error.connect(self._on_metadata_error)
        self.thread_pool.start(worker)

    def _queue_visible_thumbnails(self) -> None:
        viewport = self.list_view.viewport()
        if viewport is None:
            return

        model = self.list_view.model()
        if model is None:
            return

        viewport_rect = viewport.rect()
        top_left = self.list_view.indexAt(viewport_rect.topLeft())
        bottom_right = self.list_view.indexAt(QPoint(viewport_rect.right(), viewport_rect.bottom()))

        row = top_left.row() if top_left.isValid() else 0
        count = model.rowCount(self.list_view.rootIndex())
        if count <= 0:
            return

        if not bottom_right.isValid():
            bottom_row = min(row + 120, count - 1)
        else:
            bottom_row = bottom_right.row()

        root_index = self.list_view.rootIndex()
        for r in range(row, bottom_row + 1):
            idx = model.index(r, 0, root_index)
            if not idx.isValid():
                continue
            file_path = self._file_path_from_list_index(idx)
            if file_path:
                self._request_thumbnail(file_path, idx)

    def _prime_folder_thumbnails(self) -> None:
        model = self.list_view.model()
        if model is None:
            return

        root_index = self.list_view.rootIndex()
        row_count = model.rowCount(root_index)
        if row_count <= 0:
            return

        self._status_bar.showMessage(f"Prefetching thumbnails… {row_count} entries detected")
        max_row = min(row_count, self._initial_prefetch_count)
        for row in range(max_row):
            idx = model.index(row, 0, root_index)
            if not idx.isValid():
                continue
            file_path = self._file_path_from_list_index(idx)
            if file_path:
                self._request_thumbnail(file_path, idx)

        QTimer.singleShot(80, self._queue_visible_thumbnails)

    def _on_directory_loaded(self, path: str) -> None:
        current_root = self._current_list_root_path()
        if current_root and path == current_root:
            model = self.list_view.model()
            root_index = self.list_view.rootIndex()
            row_count = model.rowCount(root_index) if model is not None else 0

            elapsed_ms = 0.0
            if self._folder_open_started_at > 0:
                elapsed_ms = (time.perf_counter() - self._folder_open_started_at) * 1000.0

            self.cache_manager.set_folder_record(
                path,
                {
                    "item_count": row_count,
                    "last_loaded": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "last_load_ms": round(elapsed_ms, 1),
                },
            )

            self._status_bar.showMessage(f"Loaded {row_count} entries from {path} in {elapsed_ms:.0f} ms")
            self._prime_folder_thumbnails()

    def _request_thumbnail(
        self,
        file_path: str,
        index: QModelIndex | None = None,
        *,
        size: int | None = None,
        slice_request: dict | None = None,
        extra: dict | None = None,
    ) -> None:
        if Path(file_path).is_dir():
            return

        request_size = size if size is not None else self._thumbnail_size
        key = self._cache_key(file_path, request_size, slice_request=slice_request)

        cached = self.cache_manager.get(key)
        if cached is not None and request_size == self._thumbnail_size and slice_request is None:
            return

        if key in self._in_flight:
            return

        self._in_flight.add(key)
        job = ThumbnailJob(
            file_path=file_path,
            size=request_size,
            cache_key=key,
            slice_request=slice_request,
            extra=extra or {},
        )
        worker = ThumbnailWorker(self.image_engine, job)
        worker.signals.finished.connect(self._on_thumbnail_ready)
        worker.signals.error.connect(self._on_thumbnail_error)
        self.thread_pool.start(worker)

    def _on_thumbnail_ready(self, file_path: str, pixmap, metadata: dict) -> None:
        cache_key = metadata.get("cache_key", self._cache_key(file_path))
        self.cache_manager.set(cache_key, pixmap)
        self._in_flight.discard(cache_key)

        if bool(metadata.get("quicklook")):
            self._quicklook_dialog.set_pixmap(pixmap)
            self._quicklook_dialog.show()
            self._center_dialog(self._quicklook_dialog)
            return

        if metadata.get("scrub"):
            self.cache_manager.set(self._cache_key(file_path), pixmap)

        self._file_metadata[file_path] = {**self._file_metadata.get(file_path, {}), **metadata}

        source_index = self.model.index(file_path)
        proxy_index = self.list_proxy.mapFromSource(source_index)
        if proxy_index.isValid():
            rect = self.list_view.visualRect(proxy_index)
            if not rect.isNull():
                viewport = self.list_view.viewport()
                if viewport is not None:
                    viewport.update(rect)

    def _on_thumbnail_error(self, file_path: str, error_message: str) -> None:
        self._file_metadata[file_path] = {**self._file_metadata.get(file_path, {}), "broken": True, "error": error_message}
        for size in (self._thumbnail_size, self._quicklook_size):
            self._in_flight.discard(self._cache_key(file_path, size))

    def _on_metadata_ready(self, file_path: str, metadata: dict) -> None:
        self._metadata_in_flight.discard(file_path)
        self._file_metadata[file_path] = {**self._file_metadata.get(file_path, {}), **metadata}

        current = self.list_view.currentIndex()
        if current.isValid() and self._file_path_from_list_index(current) == file_path:
            self._render_metadata(file_path, self._file_metadata[file_path])

        source_index = self.model.index(file_path)
        proxy_index = self.list_proxy.mapFromSource(source_index)
        if proxy_index.isValid():
            rect = self.list_view.visualRect(proxy_index)
            if not rect.isNull():
                viewport = self.list_view.viewport()
                if viewport is not None:
                    viewport.update(rect)

    def _on_metadata_error(self, file_path: str, error_message: str) -> None:
        self._metadata_in_flight.discard(file_path)
        self._file_metadata[file_path] = {**self._file_metadata.get(file_path, {}), "broken": True, "error": error_message}

        current = self.list_view.currentIndex()
        if current.isValid() and self._file_path_from_list_index(current) == file_path:
            self._render_metadata(file_path, self._file_metadata[file_path])

    def _render_metadata(self, file_path: str, metadata: dict) -> None:
        if metadata.get("broken"):
            text = f"Path: {file_path}\n\nStatus: Broken/Unreadable\nError: {metadata.get('error', 'Unknown')}"
            self.metadata_text.setPlainText(text)
            return

        shape = metadata.get("shape_tczyx", (metadata.get("t_count", 1), metadata.get("c_count", 1), metadata.get("z_count", 1), "?", "?"))
        pps = metadata.get("pixel_size_um")
        pps_text = "N/A" if not pps else f"Z={pps.get('z')}  Y={pps.get('y')}  X={pps.get('x')}"

        text = (
            f"Path: {file_path}\n\n"
            f"Image Shape (T,C,Z,Y,X): {shape}\n"
            f"Physical Pixel Size (um): {pps_text}\n"
            f"Data Type: {metadata.get('dtype', 'Unknown')}\n"
            f"Source: {metadata.get('source', 'Unknown')}\n"
        )
        self.metadata_text.setPlainText(text)

    def _open_quicklook(self, file_path: str) -> None:
        if self._quicklook_dialog.isVisible():
            self._quicklook_dialog.close()
            return

        cached = self.cache_manager.get(self._cache_key(file_path, self._quicklook_size))
        if cached is not None:
            self._quicklook_dialog.set_pixmap(cached)
            self._quicklook_dialog.show()
            self._center_dialog(self._quicklook_dialog)
            return

        self._request_thumbnail(file_path, size=self._quicklook_size, extra={"quicklook": True})

    def _on_scrub_request(self, file_path: str, fraction: float) -> None:
        metadata = self._file_metadata.get(file_path, {})
        z_count = int(metadata.get("z_count", 1) or 1)
        t_count = int(metadata.get("t_count", 1) or 1)

        mode = ""
        count = 1
        if z_count > 1:
            mode = "z"
            count = z_count
        elif t_count > 1:
            mode = "t"
            count = t_count

        if count <= 1 or not mode:
            return

        idx = int(round(fraction * (count - 1)))
        idx = max(0, min(idx, count - 1))

        last = self._last_scrub.get(file_path)
        if last == (mode, idx):
            return
        self._last_scrub[file_path] = (mode, idx)

        self._request_thumbnail(
            file_path,
            slice_request={"mode": mode, "index": idx},
            extra={"scrub": True, "slice_mode": mode, "slice_index": idx},
        )

    def _show_context_menu(self, file_path: str, global_pos: QPoint) -> None:
        menu = QMenu(self)
        reveal_action = menu.addAction("Reveal in File Explorer")
        copy_action = menu.addAction("Copy Path")
        clear_cache_action = menu.addAction("Clear Cache for Folder")

        chosen = menu.exec(global_pos)
        if chosen is reveal_action:
            self._reveal_in_explorer(file_path)
        elif chosen is copy_action:
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(file_path)
        elif chosen is clear_cache_action:
            folder = str(Path(file_path).parent)
            self._clear_cache_for_folder(folder)

    def _reveal_in_explorer(self, file_path: str) -> None:
        try:
            subprocess.run(["explorer", "/select,", str(Path(file_path))], check=False)
        except Exception:
            pass

    def _clear_cache_for_folder(self, folder_path: str) -> None:
        normalized = str(Path(folder_path))
        self.cache_manager.delete_contains(f"::{normalized}")

        keys_to_drop = [key for key in self._file_metadata if str(Path(key).parent) == normalized]
        for key in keys_to_drop:
            self._file_metadata.pop(key, None)

        viewport = self.list_view.viewport()
        if viewport is not None:
            viewport.update()

    def _center_dialog(self, dialog: QDialog) -> None:
        parent_geo = self.geometry()
        x = parent_geo.center().x() - dialog.width() // 2
        y = parent_geo.center().y() - dialog.height() // 2
        dialog.move(x, y)

    def _file_path_from_list_index(self, index: QModelIndex) -> str:
        if not index.isValid():
            return ""
        source_index = self.list_proxy.mapToSource(index)
        if not source_index.isValid():
            return ""
        return self.model.filePath(source_index)

    def _current_list_root_path(self) -> str:
        root = self.list_view.rootIndex()
        if not root.isValid():
            return ""
        source_root = self.list_proxy.mapToSource(root)
        if not source_root.isValid():
            return ""
        return self.model.filePath(source_root)

    def closeEvent(self, a0) -> None:
        self.thread_pool.waitForDone(1500)
        self.cache_manager.close()
        super().closeEvent(a0)
