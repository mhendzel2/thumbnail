#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import h5py
except Exception:  # pragma: no cover - optional dependency
    h5py = None

try:
    import zarr
except Exception:  # pragma: no cover - optional dependency
    zarr = None

try:
    from aicsimageio import AICSImage
except Exception:  # pragma: no cover - optional dependency
    AICSImage = None

try:
    import tifffile
except Exception:  # pragma: no cover - optional dependency
    tifffile = None

try:
    import readlif  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    readlif = None

try:
    import bioformats_jar  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    bioformats_jar = None

try:
    from diskcache import Cache
except Exception:  # pragma: no cover - optional dependency
    Cache = None

from PyQt6.QtCore import (
    QByteArray,
    QBuffer,
    QDir,
    QIODevice,
    QObject,
    QRectF,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFileSystemModel,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHeaderView,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTreeView,
)

SUPPORTED_EXTENSIONS = {".ims", ".zarr", ".tif", ".tiff", ".czi", ".lif"}
THUMBNAIL_SIZE = 180
CACHE_DIR = os.path.expanduser("~/.cache/bioarena")

APP_STYLESHEET = """
QWidget {
    background-color: #2b2b2b;
    color: #e0e0e0;
    font-size: 12px;
}
QTreeView, QListWidget, QTableWidget {
    background-color: #2f2f2f;
    border: 1px solid #444444;
    alternate-background-color: #333333;
}
QTreeView::item:selected,
QTableWidget::item:selected {
    background-color: #f0a30a;
    color: #111111;
}
QListWidget::item {
    color: #ffffff;
    border: 2px solid transparent;
    padding: 6px;
    margin: 4px;
}
QListWidget::item:selected {
    border: 2px solid #f0a30a;
    background-color: #3a3a3a;
    color: #ffffff;
}
QHeaderView::section {
    background-color: #353535;
    color: #e0e0e0;
    border: 1px solid #444444;
    padding: 4px;
}
QSplitter::handle {
    background-color: #444444;
}
"""


def is_supported_path(path: str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def decode_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.bytes_):
        return value.astype(str)
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return decode_value(value.reshape(-1)[0])
        flat = [decode_value(v) for v in value.reshape(-1).tolist()]
        return ", ".join(flat)
    if isinstance(value, (list, tuple)):
        return ", ".join(decode_value(v) for v in value)
    return str(value)


def normalize_channel_to_uint8(channel: np.ndarray) -> np.ndarray:
    data = np.asarray(channel, dtype=np.float32)
    finite = np.isfinite(data)
    if not np.any(finite):
        return np.zeros(data.shape, dtype=np.uint8)
    cmin = float(np.min(data[finite]))
    cmax = float(np.max(data[finite]))
    if math.isclose(cmin, cmax):
        return np.zeros(data.shape, dtype=np.uint8)
    scaled = (data - cmin) / (cmax - cmin)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        out = np.empty(arr.shape, dtype=np.uint8)
        for c in range(arr.shape[-1]):
            out[..., c] = normalize_channel_to_uint8(arr[..., c])
        if arr.shape[-1] == 4:
            out[..., 3] = np.clip(arr[..., 3], 0, 255).astype(np.uint8)
        return out
    return normalize_channel_to_uint8(arr)


def ensure_numpy(data: Any) -> np.ndarray:
    if hasattr(data, "compute"):
        data = data.compute()
    return np.asarray(data)


def to_native_byteorder(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    byteorder = arr.dtype.byteorder
    if byteorder in ("=", "|"):
        return arr
    return arr.byteswap().view(arr.dtype.newbyteorder("="))


def squeeze_to_image(array: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(array))
    while arr.ndim > 3:
        arr = arr[0]

    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            return arr
        if arr.shape[0] in (3, 4):
            return np.moveaxis(arr, 0, -1)
        return arr[arr.shape[0] // 2]

    if arr.ndim == 2:
        return arr

    if arr.ndim == 1:
        side = int(math.sqrt(arr.size))
        if side * side == arr.size:
            return arr.reshape(side, side)
        return arr[np.newaxis, :]

    if arr.ndim == 0:
        return np.array([[arr.item()]])

    return arr


def extract_display_plane(data: Any, axes: Optional[str] = None) -> np.ndarray:
    if axes:
        axes_upper = axes.upper()
        ndim = len(getattr(data, "shape", []))
        if len(axes_upper) == ndim:
            shape = data.shape
            index: List[Any] = [slice(None)] * ndim
            preserve: List[int] = []

            if "Y" in axes_upper and "X" in axes_upper:
                preserve.extend([axes_upper.index("Y"), axes_upper.index("X")])
            if "C" in axes_upper:
                c_idx = axes_upper.index("C")
                if shape[c_idx] in (3, 4):
                    preserve.append(c_idx)

            preserve_set = set(preserve)
            for i, axis_name in enumerate(axes_upper):
                if i in preserve_set:
                    continue
                if axis_name == "Z":
                    index[i] = shape[i] // 2
                else:
                    index[i] = 0

            sliced = data[tuple(index)]
            plane = ensure_numpy(sliced)
            remaining_axes = [
                axis_name
                for i, axis_name in enumerate(axes_upper)
                if isinstance(index[i], slice)
            ]

            if "Y" in remaining_axes and "X" in remaining_axes and plane.ndim >= 2:
                transpose_order = [remaining_axes.index("Y"), remaining_axes.index("X")]
                if "C" in remaining_axes:
                    transpose_order.append(remaining_axes.index("C"))
                plane = np.transpose(plane, axes=transpose_order)

            return squeeze_to_image(plane)

    return squeeze_to_image(ensure_numpy(data))


def array_to_qimage(array: np.ndarray) -> QImage:
    arr = np.asarray(array)

    if arr.ndim == 2:
        gray = normalize_to_uint8(arr)
        h, w = gray.shape
        qimg = QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8)
        return qimg.copy()

    if arr.ndim == 3:
        if arr.shape[-1] not in (3, 4):
            arr = squeeze_to_image(arr)
            if arr.ndim == 2:
                gray = normalize_to_uint8(arr)
                h, w = gray.shape
                qimg = QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8)
                return qimg.copy()
            if arr.ndim == 3 and arr.shape[0] in (3, 4):
                arr = np.moveaxis(arr, 0, -1)

        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            rgb = normalize_to_uint8(arr)
            h, w, c = rgb.shape
            if c == 3:
                fmt = QImage.Format.Format_RGB888
                bpl = w * 3
            else:
                fmt = QImage.Format.Format_RGBA8888
                bpl = w * 4
            qimg = QImage(rgb.data, w, h, bpl, fmt)
            return qimg.copy()

    return QImage()


def fit_image_to_canvas(image: QImage, size: int = THUMBNAIL_SIZE) -> QImage:
    if image.isNull():
        return QImage()
    canvas = QImage(size, size, QImage.Format.Format_ARGB32)
    canvas.fill(QColor("#1f1f1f"))

    scaled = image.scaled(
        size - 6,
        size - 6,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )

    painter = QPainter(canvas)
    x = (size - scaled.width()) // 2
    y = (size - scaled.height()) // 2
    painter.drawImage(x, y, scaled)
    painter.end()
    return canvas


def qimage_to_png_bytes(image: QImage) -> bytes:
    if image.isNull():
        return b""
    payload = QByteArray()
    buffer = QBuffer(payload)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    buffer.close()
    return bytes(payload)


def qimage_from_png_bytes(data: bytes) -> QImage:
    image = QImage()
    if data:
        image.loadFromData(data, "PNG")
    return image


def cache_key_for_path(path: str) -> str:
    try:
        stat = os.stat(path)
        return f"{path}|{stat.st_mtime_ns}|{stat.st_size}|{THUMBNAIL_SIZE}"
    except OSError:
        return f"{path}|missing|{THUMBNAIL_SIZE}"


def collect_zarr_arrays(group: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    arrays: List[Tuple[str, Any]] = []
    for key in group.keys():
        obj = group[key]
        obj_path = f"{prefix}/{key}" if prefix else key
        if hasattr(zarr, "Array") and isinstance(obj, zarr.Array):
            arrays.append((obj_path, obj))
        elif hasattr(zarr, "Group") and isinstance(obj, zarr.Group):
            arrays.extend(collect_zarr_arrays(obj, obj_path))
    return arrays


def read_imaris_thumbnail(path: str) -> np.ndarray:
    if h5py is None:
        raise RuntimeError("h5py is required for .ims files")

    with h5py.File(path, "r") as handle:
        if "Thumbnail/Data" in handle:
            data = np.asarray(handle["Thumbnail/Data"][...])
            data = np.squeeze(data)
            if data.ndim == 1 and data.size % 4 == 0:
                dim = int(math.sqrt(data.size // 4))
                if dim * dim * 4 == data.size:
                    return data.reshape(dim, dim, 4)
            if data.ndim in (2, 3):
                return data

        dataset_root = handle.get("DataSet")
        if dataset_root is None:
            raise RuntimeError("Imaris file is missing /DataSet")

        levels = [name for name in dataset_root.keys() if "ResolutionLevel" in name]
        if not levels:
            raise RuntimeError("No resolution pyramid found in Imaris file")

        lowest_res = sorted(levels, key=lambda x: int(x.split()[-1]))[-1]
        default_path = f"DataSet/{lowest_res}/TimePoint 0/Channel 0/Data"

        if default_path in handle:
            dataset = handle[default_path]
        else:
            level_group = handle[f"DataSet/{lowest_res}"]
            timepoint_name = sorted(level_group.keys())[0]
            channel_group = level_group[timepoint_name]
            channel_name = sorted(channel_group.keys())[0]
            dataset = channel_group[channel_name]["Data"]

        if dataset.ndim >= 3:
            z_mid = dataset.shape[0] // 2
            return np.asarray(dataset[z_mid, ...])
        return np.asarray(dataset[...])


def read_zarr_thumbnail(path: str) -> np.ndarray:
    if zarr is None:
        raise RuntimeError("zarr is required for .zarr files")

    root = zarr.open_group(path, mode="r")
    multiscales = root.attrs.get("multiscales")

    axes: Optional[str] = None
    candidates: List[Tuple[str, Any]] = []

    if isinstance(multiscales, list) and multiscales:
        first = multiscales[0]
        axes_meta = first.get("axes")
        if isinstance(axes_meta, list):
            axis_names: List[str] = []
            for axis in axes_meta:
                if isinstance(axis, dict):
                    axis_names.append(str(axis.get("name", ""))[:1].upper())
                else:
                    axis_names.append(str(axis)[:1].upper())
            axes = "".join(axis_names)

        for dset in first.get("datasets", []):
            if isinstance(dset, dict) and "path" in dset:
                dset_path = dset["path"]
                try:
                    candidates.append((dset_path, root[dset_path]))
                except Exception:
                    continue

    if not candidates:
        candidates = collect_zarr_arrays(root)

    if not candidates:
        raise RuntimeError("No arrays found in zarr group")

    _, smallest = min(candidates, key=lambda item: int(np.prod(item[1].shape)))
    if axes and len(axes) != len(smallest.shape):
        axes = None

    return extract_display_plane(smallest, axes)


def read_aics_thumbnail(path: str) -> np.ndarray:
    if AICSImage is None:
        raise RuntimeError("aicsimageio is required for this format")

    image = AICSImage(path)
    axes = getattr(image.dims, "order", None)
    return extract_display_plane(image.dask_data, axes)


def lif_dependency_hint() -> str:
    return (
        "LIF support requires `readlif>=0.6.4` "
        "(and optionally `bioformats_jar` for Bio-Formats fallback). "
        "Run install.bat."
    )


def read_lif_thumbnail(path: str) -> np.ndarray:
    if AICSImage is None:
        raise RuntimeError("aicsimageio is required for .lif files")

    try:
        return read_aics_thumbnail(path)
    except Exception as exc:
        detail = str(exc)
        if "readlif is required" in detail or "bioformats_jar is required" in detail:
            raise RuntimeError(lif_dependency_hint()) from exc
        raise


def read_tiff_thumbnail(path: str) -> np.ndarray:
    if tifffile is None:
        return read_aics_thumbnail(path)

    with tifffile.TiffFile(path) as handle:
        if not handle.series:
            raise RuntimeError("No TIFF series found")

        series = handle.series[0]
        chosen = series
        levels = getattr(series, "levels", None)
        if levels:
            chosen = min(
                levels,
                key=lambda lvl: int(np.prod(getattr(lvl, "shape", (1,)))),
            )

        axes = getattr(chosen, "axes", None) or getattr(series, "axes", None)

        pages = getattr(chosen, "pages", None)
        try:
            if pages is not None and len(pages) > 0:
                plane_idx = len(pages) // 2
                plane = pages[plane_idx].asarray()
            else:
                plane = chosen.asarray()
        except AttributeError as exc:
            # NumPy 2.x removed ndarray.newbyteorder; older tifffile releases
            # can still call it for some byte orders. Fall back to memmap path.
            if "newbyteorder" not in str(exc):
                raise
            mem = tifffile.memmap(path)
            plane = extract_display_plane(mem, getattr(series, "axes", None))
            return to_native_byteorder(np.asarray(plane))

        plane = to_native_byteorder(np.asarray(plane))
        return extract_display_plane(plane, axes)


def read_imaris_metadata(path: str) -> List[Tuple[str, str]]:
    if h5py is None:
        raise RuntimeError("h5py is required for .ims metadata")

    rows: List[Tuple[str, str]] = [
        ("Format", "Imaris (.ims)"),
        ("Path", path),
    ]

    with h5py.File(path, "r") as handle:
        image_info = handle.get("/DataSetInfo/Image")
        values: Dict[str, str] = {}

        if image_info is not None:
            for key, value in image_info.attrs.items():
                values[key] = decode_value(value)
            for key in image_info.keys():
                try:
                    node = image_info[key]
                    if hasattr(node, "shape"):
                        values[key] = decode_value(node[()])
                except Exception:
                    continue

        extent_keys = [
            "ExtMin0",
            "ExtMin1",
            "ExtMin2",
            "ExtMax0",
            "ExtMax1",
            "ExtMax2",
        ]
        if all(k in values for k in extent_keys):
            extents = (
                f"min=({values['ExtMin0']}, {values['ExtMin1']}, {values['ExtMin2']}), "
                f"max=({values['ExtMax0']}, {values['ExtMax1']}, {values['ExtMax2']})"
            )
            rows.append(("Extents", extents))
        else:
            fallback_extent = [f"{k}={v}" for k, v in values.items() if k.startswith("Ext")]
            if fallback_extent:
                rows.append(("Extents", ", ".join(sorted(fallback_extent))))

        unit = values.get("Unit") or values.get("UnitLabel")
        if unit:
            rows.append(("Unit", unit))

        size_keys = [
            k
            for k in ("X", "Y", "Z", "C", "T", "SizeX", "SizeY", "SizeZ", "SizeC", "SizeT")
            if k in values
        ]
        if size_keys:
            rows.append(("Size", ", ".join(f"{k}={values[k]}" for k in size_keys)))

        if "DataSet" in handle:
            levels = [name for name in handle["DataSet"].keys() if "ResolutionLevel" in name]
            if levels:
                rows.append(("Resolution Levels", str(len(levels))))

    return rows


def read_zarr_metadata(path: str) -> List[Tuple[str, str]]:
    if zarr is None:
        raise RuntimeError("zarr is required for .zarr metadata")

    rows: List[Tuple[str, str]] = [
        ("Format", "OME-Zarr (.zarr)"),
        ("Path", path),
    ]

    root = zarr.open_group(path, mode="r")
    multiscales = root.attrs.get("multiscales")

    if isinstance(multiscales, list):
        rows.append(("Multiscales", str(len(multiscales))))
        if multiscales:
            first = multiscales[0]
            axes = first.get("axes")
            if axes:
                axis_str = ", ".join(
                    str(axis.get("name", axis)) if isinstance(axis, dict) else str(axis)
                    for axis in axes
                )
                rows.append(("Axes", axis_str))

    arrays = collect_zarr_arrays(root)
    if arrays:
        _, smallest = min(arrays, key=lambda item: int(np.prod(item[1].shape)))
        rows.append(("Lowest Scale Shape", str(tuple(smallest.shape))))
        rows.append(("Lowest Scale DType", str(smallest.dtype)))

    return rows


def read_aics_metadata(path: str) -> List[Tuple[str, str]]:
    if AICSImage is None:
        raise RuntimeError("aicsimageio is required for metadata")

    image = AICSImage(path)
    dims = image.dims
    order = getattr(dims, "order", "")
    shape = getattr(dims, "shape", "")

    rows: List[Tuple[str, str]] = [
        ("Format", f"{Path(path).suffix.lower()}"),
        ("Path", path),
        ("Dimensions", f"{order}: {shape}"),
    ]

    physical = getattr(image, "physical_pixel_sizes", None)
    if physical is not None:
        rows.append(
            (
                "Physical Pixel Sizes",
                f"X={getattr(physical, 'X', None)}, Y={getattr(physical, 'Y', None)}, Z={getattr(physical, 'Z', None)}",
            )
        )

    return rows


def read_lif_metadata(path: str) -> List[Tuple[str, str]]:
    if AICSImage is None:
        raise RuntimeError("aicsimageio is required for .lif metadata")

    try:
        return read_aics_metadata(path)
    except Exception as exc:
        detail = str(exc)
        if "readlif is required" in detail or "bioformats_jar is required" in detail:
            raise RuntimeError(lif_dependency_hint()) from exc
        raise


def read_tiff_metadata(path: str) -> List[Tuple[str, str]]:
    if tifffile is None:
        return read_aics_metadata(path)

    rows: List[Tuple[str, str]] = [
        ("Format", "TIFF (.tif/.tiff)"),
        ("Path", path),
    ]

    with tifffile.TiffFile(path) as handle:
        rows.append(("Series Count", str(len(handle.series))))

        if handle.series:
            series = handle.series[0]
            rows.append(("Dimensions", f"{series.axes}: {series.shape}"))
            rows.append(("Data Type", str(series.dtype)))
            levels = getattr(series, "levels", None)
            if levels and len(levels) > 1:
                rows.append(("Pyramid Levels", str(len(levels))))

        if handle.pages:
            page = handle.pages[0]
            rows.append(("Page Shape", str(page.shape)))
            rows.append(("Page Type", str(page.dtype)))

    return rows


def read_folder_metadata(path: str) -> List[Tuple[str, str]]:
    stat = os.stat(path)
    return [
        ("Type", "Folder"),
        ("Path", path),
        ("Modified", str(stat.st_mtime)),
    ]


def create_folder_icon(size: int = THUMBNAIL_SIZE) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    tab = QRectF(size * 0.18, size * 0.22, size * 0.34, size * 0.16)
    body = QRectF(size * 0.12, size * 0.34, size * 0.76, size * 0.46)

    painter.setPen(QPen(QColor("#b57a00"), 2))
    painter.setBrush(QBrush(QColor("#f0c24b")))
    painter.drawRoundedRect(tab, 7, 7)
    painter.setBrush(QBrush(QColor("#f0a30a")))
    painter.drawRoundedRect(body, 9, 9)

    painter.end()
    return QIcon(pixmap)


def create_loading_icon(size: int = THUMBNAIL_SIZE) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor("#1f1f1f"))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor("#666666"), 2))
    painter.setBrush(QBrush(QColor("#2a2a2a")))
    painter.drawRoundedRect(4, 4, size - 8, size - 8, 10, 10)
    painter.setPen(QPen(QColor("#d0d0d0"), 1))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "Loading...")
    painter.end()
    return QIcon(pixmap)


def create_error_icon(size: int = THUMBNAIL_SIZE) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor("#1f1f1f"))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor("#d9534f"), 3))
    painter.drawLine(18, 18, size - 18, size - 18)
    painter.drawLine(size - 18, 18, 18, size - 18)
    painter.end()
    return QIcon(pixmap)


class BioFileModel(QFileSystemModel):
    def hasChildren(self, index):  # type: ignore[override]
        path = self.filePath(index)
        if path.lower().endswith(".zarr"):
            return False
        return super().hasChildren(index)


class DirectoryScanSignals(QObject):
    finished = pyqtSignal(int, str, object)
    error = pyqtSignal(int, str, str)


class DirectoryScanWorker(QRunnable):
    def __init__(self, request_id: int, folder_path: str) -> None:
        super().__init__()
        self.request_id = request_id
        self.folder_path = folder_path
        self.signals = DirectoryScanSignals()

    def run(self) -> None:
        try:
            entries: List[Dict[str, Any]] = []
            with os.scandir(self.folder_path) as it:
                for entry in it:
                    name = entry.name
                    full_path = entry.path

                    try:
                        is_dir = entry.is_dir(follow_symlinks=False)
                    except OSError:
                        continue

                    suffix = Path(name).suffix.lower()
                    if is_dir:
                        if suffix == ".zarr":
                            entries.append({"name": name, "path": full_path, "is_dir": False})
                        else:
                            entries.append({"name": name, "path": full_path, "is_dir": True})
                    elif suffix in SUPPORTED_EXTENSIONS:
                        entries.append({"name": name, "path": full_path, "is_dir": False})

            entries.sort(key=lambda x: (0 if x["is_dir"] else 1, x["name"].lower()))
            self.signals.finished.emit(self.request_id, self.folder_path, entries)
        except Exception as exc:
            self.signals.error.emit(self.request_id, self.folder_path, f"{type(exc).__name__}: {exc}")


class ThumbnailSignals(QObject):
    finished = pyqtSignal(int, str, QImage)
    error = pyqtSignal(int, str, str)


class ThumbnailWorker(QRunnable):
    def __init__(self, request_id: int, file_path: str, cache_dir: str) -> None:
        super().__init__()
        self.request_id = request_id
        self.file_path = file_path
        self.cache_dir = cache_dir
        self.signals = ThumbnailSignals()

    def run(self) -> None:
        try:
            key = cache_key_for_path(self.file_path)

            if Cache is not None:
                with Cache(self.cache_dir) as cache:
                    payload = cache.get(key)
                if payload:
                    cached = qimage_from_png_bytes(payload)
                    if not cached.isNull():
                        self.signals.finished.emit(self.request_id, self.file_path, cached)
                        return

            suffix = Path(self.file_path).suffix.lower()
            if suffix == ".ims":
                raw = read_imaris_thumbnail(self.file_path)
            elif suffix == ".zarr":
                raw = read_zarr_thumbnail(self.file_path)
            elif suffix in {".tif", ".tiff"}:
                raw = read_tiff_thumbnail(self.file_path)
            elif suffix == ".lif":
                raw = read_lif_thumbnail(self.file_path)
            elif suffix == ".czi":
                raw = read_aics_thumbnail(self.file_path)
            else:
                raise RuntimeError(f"Unsupported format: {suffix}")

            image = array_to_qimage(raw)
            image = fit_image_to_canvas(image, THUMBNAIL_SIZE)

            if not image.isNull() and Cache is not None:
                payload = qimage_to_png_bytes(image)
                if payload:
                    with Cache(self.cache_dir) as cache:
                        cache.set(key, payload)

            self.signals.finished.emit(self.request_id, self.file_path, image)
        except Exception as exc:
            self.signals.error.emit(self.request_id, self.file_path, f"{type(exc).__name__}: {exc}")


class MetadataSignals(QObject):
    finished = pyqtSignal(int, str, object)
    error = pyqtSignal(int, str, str)


class MetadataWorker(QRunnable):
    def __init__(self, request_id: int, path: str, is_dir: bool) -> None:
        super().__init__()
        self.request_id = request_id
        self.path = path
        self.is_dir = is_dir
        self.signals = MetadataSignals()

    def run(self) -> None:
        try:
            suffix = Path(self.path).suffix.lower()
            if self.is_dir:
                rows = read_folder_metadata(self.path)
            elif suffix == ".ims":
                rows = read_imaris_metadata(self.path)
            elif suffix == ".zarr":
                rows = read_zarr_metadata(self.path)
            elif suffix in {".tif", ".tiff"}:
                rows = read_tiff_metadata(self.path)
            elif suffix == ".lif":
                rows = read_lif_metadata(self.path)
            elif suffix == ".czi":
                rows = read_aics_metadata(self.path)
            else:
                rows = [("Path", self.path), ("Type", "Unknown")]

            self.signals.finished.emit(self.request_id, self.path, rows)
        except Exception as exc:
            self.signals.error.emit(self.request_id, self.path, f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BioArena")
        self.resize(1600, 920)

        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(max(4, os.cpu_count() or 4))

        self.active_request_id = 0
        self.metadata_request_id = 0
        self.pending_thumbnails = 0
        self.item_by_path: Dict[str, QListWidgetItem] = {}

        self.folder_icon = create_folder_icon()
        self.loading_icon = create_loading_icon()
        self.error_icon = create_error_icon()

        self._build_ui()
        self._apply_theme()
        self._show_dependency_warnings()

        self._configure_tree_root()

        start_path = str(Path.home())
        self.load_directory(start_path)

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        self.tree_model = BioFileModel(self)
        self.tree_model.setFilter(
            QDir.Filter.AllDirs | QDir.Filter.NoDotAndDotDot | QDir.Filter.Files
        )
        self.tree_model.setNameFilterDisables(False)
        self.tree_model.setNameFilters(["*.ims", "*.zarr", "*.tif", "*.tiff", "*.czi", "*.lif"])

        self.tree_view = QTreeView(splitter)
        self.tree_view.setModel(self.tree_model)
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setUniformRowHeights(True)
        self.tree_view.setAnimated(False)
        self.tree_view.clicked.connect(self._on_tree_clicked)

        for col in (1, 2, 3):
            self.tree_view.hideColumn(col)

        self.arena_view = QListWidget(splitter)
        self.arena_view.setViewMode(QListView.ViewMode.IconMode)
        self.arena_view.setResizeMode(QListView.ResizeMode.Adjust)
        self.arena_view.setMovement(QListView.Movement.Static)
        self.arena_view.setIconSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        self.arena_view.setGridSize(QSize(210, 240))
        self.arena_view.setSpacing(8)
        self.arena_view.setWordWrap(True)
        self.arena_view.setWrapping(True)
        self.arena_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.arena_view.itemClicked.connect(self._on_arena_item_clicked)
        self.arena_view.itemDoubleClicked.connect(self._on_arena_item_double_clicked)

        self.metadata_table = QTableWidget(splitter)
        self.metadata_table.setColumnCount(2)
        self.metadata_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.metadata_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.metadata_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.metadata_table.verticalHeader().setVisible(False)
        self.metadata_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.metadata_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        splitter.addWidget(self.tree_view)
        splitter.addWidget(self.arena_view)
        splitter.addWidget(self.metadata_table)
        splitter.setSizes([300, 980, 320])

        self.setCentralWidget(splitter)
        self.setStatusBar(QStatusBar(self))

    def _apply_theme(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(APP_STYLESHEET)

    def _configure_tree_root(self) -> None:
        # On Windows, an empty root path exposes all drive letters.
        if os.name == "nt":
            root_path = ""
            self.tree_model.setRootPath(root_path)
            self.tree_view.setRootIndex(self.tree_model.index(root_path))
            return

        root_path = QDir.rootPath()
        self.tree_model.setRootPath(root_path)
        self.tree_view.setRootIndex(self.tree_model.index(root_path))

    def _show_dependency_warnings(self) -> None:
        missing = []
        if h5py is None:
            missing.append("h5py")
        if zarr is None:
            missing.append("zarr")
        if AICSImage is None:
            missing.append("aicsimageio")
        if tifffile is None:
            missing.append("tifffile")
        if readlif is None:
            missing.append("readlif (LIF)")
        if bioformats_jar is None:
            missing.append("bioformats_jar (Bio-Formats)")
        if Cache is None:
            missing.append("diskcache")

        if missing:
            self.statusBar().showMessage(
                "Missing optional dependencies: " + ", ".join(missing)
            )

    def _on_tree_clicked(self, index) -> None:
        path = self.tree_model.filePath(index)
        is_dir = self.tree_model.isDir(index)

        if is_dir and not path.lower().endswith(".zarr"):
            self.load_directory(path)
        else:
            self.show_single_item(path, is_dir=False)

    def load_directory(self, folder_path: str) -> None:
        self.active_request_id += 1
        request_id = self.active_request_id
        self.pending_thumbnails = 0

        self.arena_view.clear()
        self.item_by_path.clear()
        self._set_metadata_rows([])

        loading = QListWidgetItem("Loading...")
        loading.setIcon(self.loading_icon)
        loading.setFlags(Qt.ItemFlag.NoItemFlags)
        loading.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
        self.arena_view.addItem(loading)

        worker = DirectoryScanWorker(request_id, folder_path)
        worker.signals.finished.connect(self._on_directory_scanned)
        worker.signals.error.connect(self._on_directory_scan_error)
        self.thread_pool.start(worker)

        self.statusBar().showMessage(f"Scanning: {folder_path}")

    def show_single_item(self, path: str, is_dir: bool) -> None:
        self.active_request_id += 1
        request_id = self.active_request_id
        self.pending_thumbnails = 0

        self.arena_view.clear()
        self.item_by_path.clear()
        self._set_metadata_rows([])

        name = Path(path).name or path
        item = QListWidgetItem(name)
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setData(Qt.ItemDataRole.UserRole + 1, is_dir)
        item.setData(Qt.ItemDataRole.UserRole + 2, name)
        item.setTextAlignment(int(Qt.AlignmentFlag.AlignHCenter))

        if is_dir:
            item.setIcon(self.folder_icon)
        elif is_supported_path(path):
            item.setIcon(self.loading_icon)
            item.setText(f"{name}\nLoading...")
            self.pending_thumbnails = 1
            self._queue_thumbnail(request_id, path)
        else:
            item.setIcon(self.error_icon)

        self.arena_view.addItem(item)
        self.item_by_path[path] = item
        self.statusBar().showMessage(f"Selected: {path}")

    def _on_directory_scanned(self, request_id: int, folder_path: str, entries: Sequence[Dict[str, Any]]) -> None:
        if request_id != self.active_request_id:
            return

        self.arena_view.clear()
        self.item_by_path.clear()
        self.pending_thumbnails = 0

        if not entries:
            empty = QListWidgetItem("No folders or supported files found")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self.arena_view.addItem(empty)
            self.statusBar().showMessage(f"No displayable items in: {folder_path}")
            return

        for entry in entries:
            name = entry["name"]
            path = entry["path"]
            is_dir = bool(entry["is_dir"])

            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setData(Qt.ItemDataRole.UserRole + 1, is_dir)
            item.setData(Qt.ItemDataRole.UserRole + 2, name)
            item.setTextAlignment(int(Qt.AlignmentFlag.AlignHCenter))

            if is_dir:
                item.setIcon(self.folder_icon)
            else:
                item.setIcon(self.loading_icon)
                item.setText(f"{name}\nLoading...")
                self.pending_thumbnails += 1
                self._queue_thumbnail(request_id, path)

            self.arena_view.addItem(item)
            self.item_by_path[path] = item

        self.statusBar().showMessage(
            f"Loaded {len(entries)} items from {folder_path}. Generating {self.pending_thumbnails} thumbnails..."
        )

    def _on_directory_scan_error(self, request_id: int, folder_path: str, error: str) -> None:
        if request_id != self.active_request_id:
            return
        self.arena_view.clear()
        failed = QListWidgetItem("Unable to read folder")
        failed.setFlags(Qt.ItemFlag.NoItemFlags)
        self.arena_view.addItem(failed)
        self.statusBar().showMessage(f"Directory scan failed: {error}")
        QMessageBox.warning(self, "BioArena", f"Failed to scan folder:\n{folder_path}\n\n{error}")

    def _queue_thumbnail(self, request_id: int, path: str) -> None:
        worker = ThumbnailWorker(request_id, path, CACHE_DIR)
        worker.signals.finished.connect(self._on_thumbnail_ready)
        worker.signals.error.connect(self._on_thumbnail_error)
        self.thread_pool.start(worker)

    def _on_thumbnail_ready(self, request_id: int, path: str, image: QImage) -> None:
        if request_id != self.active_request_id:
            return

        item = self.item_by_path.get(path)
        if item is None:
            return

        name = item.data(Qt.ItemDataRole.UserRole + 2) or Path(path).name
        if image.isNull():
            item.setIcon(self.error_icon)
            item.setText(f"{name}\nUnavailable")
        else:
            item.setIcon(QIcon(QPixmap.fromImage(image)))
            item.setText(str(name))

        if self.pending_thumbnails > 0:
            self.pending_thumbnails -= 1
        if self.pending_thumbnails == 0:
            self.statusBar().showMessage("Thumbnail generation complete")

    def _on_thumbnail_error(self, request_id: int, path: str, error: str) -> None:
        if request_id != self.active_request_id:
            return

        item = self.item_by_path.get(path)
        if item is not None:
            name = item.data(Qt.ItemDataRole.UserRole + 2) or Path(path).name
            item.setIcon(self.error_icon)
            item.setText(f"{name}\nUnavailable")

        if self.pending_thumbnails > 0:
            self.pending_thumbnails -= 1

        self.statusBar().showMessage(f"Thumbnail failed for {Path(path).name}: {error}")

    def _on_arena_item_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        is_dir = bool(item.data(Qt.ItemDataRole.UserRole + 1))
        if not path:
            return

        self.metadata_request_id += 1
        request_id = self.metadata_request_id

        worker = MetadataWorker(request_id, str(path), is_dir)
        worker.signals.finished.connect(self._on_metadata_ready)
        worker.signals.error.connect(self._on_metadata_error)
        self.thread_pool.start(worker)

        self.statusBar().showMessage(f"Loading metadata: {path}")

    def _on_arena_item_double_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        is_dir = bool(item.data(Qt.ItemDataRole.UserRole + 1))
        if path and is_dir:
            self.load_directory(str(path))

    def _on_metadata_ready(self, request_id: int, path: str, rows: Sequence[Tuple[str, str]]) -> None:
        if request_id != self.metadata_request_id:
            return
        self._set_metadata_rows(rows)
        self.statusBar().showMessage(f"Metadata loaded: {path}")

    def _on_metadata_error(self, request_id: int, path: str, error: str) -> None:
        if request_id != self.metadata_request_id:
            return
        self._set_metadata_rows(
            [
                ("Path", path),
                ("Error", error),
            ]
        )
        self.statusBar().showMessage(f"Metadata failed: {error}")

    def _set_metadata_rows(self, rows: Sequence[Tuple[str, str]]) -> None:
        self.metadata_table.clearContents()
        self.metadata_table.setRowCount(len(rows))

        for r, (key, value) in enumerate(rows):
            key_item = QTableWidgetItem(str(key))
            value_item = QTableWidgetItem(str(value))
            key_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            value_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.metadata_table.setItem(r, 0, key_item)
            self.metadata_table.setItem(r, 1, value_item)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
