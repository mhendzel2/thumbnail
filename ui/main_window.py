from __future__ import annotations

from pathlib import Path
import subprocess
import time

from PyQt6.QtCore import QDir, QModelIndex, QPoint, QThreadPool, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFileSystemModel
from PyQt6.QtWidgets import (
    QApplication,
    QMenu,
    QFileIconProvider,
    QDialog,
    QDockWidget,
    QLabel,
    QListView,
    QMainWindow,
    QPlainTextEdit,
    QStatusBar,
    QSplitter,
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
    "*.ims",
    "*.czi",
    "*.nd2",
    "*.tif",
    "*.tiff",
    "*.ome.tif",
    "*.ome.tiff",
]


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
                model = self.model()
                if isinstance(model, QFileSystemModel):
                    self.quickLookRequested.emit(model.filePath(idx))
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

        model = self.model()
        if not isinstance(model, QFileSystemModel):
            return

        rect = self.visualRect(idx)
        if rect.width() <= 0:
            return

        fraction = (e.position().x() - rect.left()) / rect.width()
        fraction = min(1.0, max(0.0, float(fraction)))
        self.scrubRequested.emit(model.filePath(idx), fraction)

    def contextMenuEvent(self, a0) -> None:
        if a0 is None:
            return

        idx = self.indexAt(a0.pos())
        if not idx.isValid():
            return

        model = self.model()
        if not isinstance(model, QFileSystemModel):
            return

        self.setCurrentIndex(idx)
        self.contextMenuRequested.emit(model.filePath(idx), a0.globalPos())


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

        self.tree_view = self._build_tree_view()
        self.list_view = self._build_list_view()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.tree_view)
        splitter.addWidget(self.list_view)
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
        self._status_bar.showMessage(
            f"Ready — folder cache database: {self.cache_manager.cache_db_path()}"
        )

        self._setup_metadata_dock()

        fs_root_index = QModelIndex()

        self.tree_view.setRootIndex(fs_root_index)
        self.list_view.setRootIndex(fs_root_index)

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
        view.setModel(self.model)
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
            self._status_bar.showMessage(
                f"Opening folder: {path} (seen before, last items: {prev_count}, at {prev_time})"
            )
        else:
            self._status_bar.showMessage(f"Opening folder: {path} (first load)")

        new_root = self.model.index(path)
        self.list_view.setRootIndex(new_root)
        self._prime_folder_thumbnails()
        self._queue_visible_thumbnails()

    def _on_list_current_changed(self, current: QModelIndex, previous: QModelIndex) -> None:
        if not current.isValid():
            return

        file_path = self.model.filePath(current)
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
        viewport_rect = viewport.rect()
        top_left = self.list_view.indexAt(viewport_rect.topLeft())
        bottom_right = self.list_view.indexAt(QPoint(viewport_rect.right(), viewport_rect.bottom()))
        row = top_left.row() if top_left.isValid() else 0
        if not bottom_right.isValid():
            bottom_row = min(row + 120, self.model.rowCount(self.list_view.rootIndex()) - 1)
        else:
            bottom_row = bottom_right.row()

        root_index = self.list_view.rootIndex()
        for r in range(row, bottom_row + 1):
            idx = self.model.index(r, 0, root_index)
            if not idx.isValid():
                continue
            self._request_thumbnail(self.model.filePath(idx), idx)

    def _prime_folder_thumbnails(self) -> None:
        root_index = self.list_view.rootIndex()
        row_count = self.model.rowCount(root_index)
        if row_count <= 0:
            return

        self._status_bar.showMessage(f"Prefetching thumbnails… {row_count} entries detected")

        max_row = min(row_count, self._initial_prefetch_count)
        for row in range(max_row):
            idx = self.model.index(row, 0, root_index)
            if idx.isValid():
                self._request_thumbnail(self.model.filePath(idx), idx)

        QTimer.singleShot(80, self._queue_visible_thumbnails)

    def _on_directory_loaded(self, path: str) -> None:
        current_root = self.model.filePath(self.list_view.rootIndex())
        if current_root and path == current_root:
            root_index = self.list_view.rootIndex()
            row_count = self.model.rowCount(root_index)
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

            self._status_bar.showMessage(
                f"Loaded {row_count} entries from {path} in {elapsed_ms:.0f} ms"
            )
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

        is_quicklook = bool(metadata.get("quicklook"))
        if is_quicklook:
            self._quicklook_dialog.set_pixmap(pixmap)
            self._quicklook_dialog.show()
            self._center_dialog(self._quicklook_dialog)
            return

        if metadata.get("scrub"):
            self.cache_manager.set(self._cache_key(file_path), pixmap)

        self._file_metadata[file_path] = {**self._file_metadata.get(file_path, {}), **metadata}

        index = self.model.index(file_path)
        if index.isValid():
            rect = self.list_view.visualRect(index)
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
        if current.isValid() and self.model.filePath(current) == file_path:
            self._render_metadata(file_path, self._file_metadata[file_path])

        idx = self.model.index(file_path)
        if idx.isValid():
            rect = self.list_view.visualRect(idx)
            if not rect.isNull():
                viewport = self.list_view.viewport()
                if viewport is not None:
                    viewport.update(rect)

    def _on_metadata_error(self, file_path: str, error_message: str) -> None:
        self._metadata_in_flight.discard(file_path)
        self._file_metadata[file_path] = {**self._file_metadata.get(file_path, {}), "broken": True, "error": error_message}

        current = self.list_view.currentIndex()
        if current.isValid() and self.model.filePath(current) == file_path:
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

        self._request_thumbnail(
            file_path,
            size=self._quicklook_size,
            extra={"quicklook": True},
        )

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

    def closeEvent(self, a0) -> None:
        self.thread_pool.waitForDone(1500)
        self.cache_manager.close()
        super().closeEvent(a0)
