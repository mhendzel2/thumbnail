from __future__ import annotations

import io
import importlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap

try:
    import tifffile
except Exception:
    tifffile = None

h5py = None

LOGGER = logging.getLogger(__name__)

STANDARD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".psd"}
MICROSCOPY_EXTENSIONS = {".ims", ".czi", ".nd2", ".stk", ".tiff", ".tif", ".ome.tif", ".ome.tiff", ".mvd2", ".acff"}


@dataclass(slots=True)
class ThumbnailResult:
    pixmap: QPixmap
    metadata: dict[str, Any]


class ImageEngine:
    def __init__(self) -> None:
        self._max_tiff_plane_pixels = 40_000_000
        self._max_ims_plane_pixels = 24_000_000
        self._explicit_bioio_readers = {
            ".czi": "bioio_czi",
            ".nd2": "bioio_nd2",
        }

    def _get_h5py(self):
        global h5py
        if h5py is None:
            try:
                h5py = importlib.import_module("h5py")
            except Exception:
                return None
        return h5py

    def load_thumbnail(
        self,
        file_path: str,
        size: int,
        *,
        slice_request: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> ThumbnailResult:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in STANDARD_EXTENSIONS:
            return self._load_standard_thumbnail(path, size)

        if self._is_ims_path(path):
            ims_result = self._load_ims_thumbnail(path, size, slice_request=slice_request)
            if ims_result is not None:
                return ims_result
            return ThumbnailResult(
                pixmap=self._broken_placeholder(size),
                metadata={"broken": True, "error": "IMS decode unavailable", "source": "ims-hdf5"},
            )

        if self._is_tiff_path(path) and not self._is_ome_tiff_path(path):
            fallback = self._load_tiff_thumbnail(path, size, slice_request=slice_request)
            if fallback is not None:
                return fallback
            return ThumbnailResult(
                pixmap=self._broken_placeholder(size),
                metadata={"broken": True, "error": "TIFF decode unavailable", "source": "tifffile-fallback"},
            )

        if self._is_microscopy_path(path):
            return self._load_microscopy_thumbnail(path, size, slice_request=slice_request)

        return ThumbnailResult(
            pixmap=self._broken_placeholder(size),
            metadata={"broken": True, "error": "Unsupported file format"},
        )

    def load_metadata(self, file_path: str) -> dict[str, Any]:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in STANDARD_EXTENSIONS:
            try:
                with Image.open(path) as image:
                    channels = len(image.getbands())
                    y, x = image.height, image.width
                    return {
                        "shape_tczyx": (1, channels, 1, y, x),
                        "dtype": str(np.asarray(image).dtype),
                        "pixel_size_um": None,
                        "t_count": 1,
                        "c_count": channels,
                        "z_count": 1,
                    }
            except Exception as exc:
                return {"broken": True, "error": str(exc)}

        if self._is_ims_path(path):
            ims_meta = self._load_ims_metadata(path)
            if ims_meta is not None:
                ims_meta["source"] = "ims-hdf5"
                return ims_meta
            return {"broken": True, "error": "IMS metadata unavailable", "source": "ims-hdf5"}

        if self._is_tiff_path(path) and not self._is_ome_tiff_path(path):
            fallback = self._load_tiff_metadata(path)
            if fallback is not None:
                fallback["source"] = "tifffile-fallback"
                return fallback
            return {"broken": True, "error": "TIFF metadata unavailable", "source": "tifffile-fallback"}

        if self._is_microscopy_path(path):
            try:
                bio_image = self._open_bioimage(path)
                metadata = self._collect_bio_metadata(bio_image)
                return metadata
            except Exception as exc:
                if self._is_tiff_path(path):
                    fallback = self._load_tiff_metadata(path)
                    if fallback is not None:
                        fallback["source"] = "tifffile-fallback"
                        fallback["bioio_error"] = str(exc)
                        return fallback

                LOGGER.warning("Failed to read metadata for %s: %s", file_path, exc)
                return {"broken": True, "error": str(exc)}

        return {"broken": True, "error": "Unsupported file format"}

    def _is_microscopy_path(self, path: Path) -> bool:
        name = path.name.lower()
        return name.endswith(".ome.tif") or name.endswith(".ome.tiff") or path.suffix.lower() in MICROSCOPY_EXTENSIONS

    def _is_tiff_path(self, path: Path) -> bool:
        name = path.name.lower()
        return name.endswith(".tif") or name.endswith(".tiff") or name.endswith(".stk")

    def _is_ims_path(self, path: Path) -> bool:
        return path.name.lower().endswith(".ims")

    def _is_ome_tiff_path(self, path: Path) -> bool:
        name = path.name.lower()
        return name.endswith(".ome.tif") or name.endswith(".ome.tiff")

    def _load_standard_thumbnail(self, path: Path, size: int) -> ThumbnailResult:
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)

                exif_thumb = self._extract_exif_thumbnail(image)
                if exif_thumb is not None:
                    exif_thumb.thumbnail((size, size), Image.Resampling.LANCZOS)
                    pixmap = self._pil_to_fitted_qpixmap(exif_thumb, size)
                    return ThumbnailResult(pixmap=pixmap, metadata={"source": "exif", "t_count": 1, "z_count": 1, "c_count": len(image.getbands())})

                image.thumbnail((size, size), Image.Resampling.LANCZOS)
                pixmap = self._pil_to_fitted_qpixmap(image, size)
                return ThumbnailResult(pixmap=pixmap, metadata={"source": "resized", "t_count": 1, "z_count": 1, "c_count": len(image.getbands())})
        except Exception as exc:
            LOGGER.exception("Failed to read standard image %s", path)
            return ThumbnailResult(
                pixmap=self._broken_placeholder(size),
                metadata={"broken": True, "error": str(exc)},
            )

    def _load_microscopy_thumbnail(
        self,
        path: Path,
        size: int,
        *,
        slice_request: dict[str, Any] | None = None,
    ) -> ThumbnailResult:
        try:
            bio_image = self._open_bioimage(path)
            self._select_lowest_resolution_if_available(bio_image)
            metadata = self._collect_bio_metadata(bio_image)

            t_index = 0
            if metadata.get("t_count", 1) > 1 and slice_request and slice_request.get("mode") == "t":
                t_index = int(slice_request.get("index", 0))

            data = self._read_tczyx_data(bio_image, t_index=t_index)
            processed = self._reduce_tczyx_to_display(data, metadata, slice_request=slice_request)
            contrasted = self._auto_contrast(processed)
            pixmap = self._numpy_to_fitted_qpixmap(contrasted, size)

            metadata["source"] = "bioio"
            return ThumbnailResult(pixmap=pixmap, metadata=metadata)
        except Exception as exc:
            if self._is_tiff_path(path):
                fallback = self._load_tiff_thumbnail(path, size, slice_request=slice_request)
                if fallback is not None:
                    fallback.metadata["bioio_error"] = str(exc)
                    return fallback

            LOGGER.warning("Failed microscopy decode for %s: %s", path, exc)
            return ThumbnailResult(
                pixmap=self._broken_placeholder(size),
                metadata={"broken": True, "error": str(exc)},
            )

    def _load_tiff_thumbnail(
        self,
        path: Path,
        size: int,
        *,
        slice_request: dict[str, Any] | None = None,
    ) -> ThumbnailResult | None:
        try:
            with Image.open(path) as image:
                image.thumbnail((size, size), Image.Resampling.BILINEAR)
                pixmap = self._pil_to_fitted_qpixmap(image, size)
                metadata = self._load_tiff_metadata(path) or {
                    "shape_tczyx": (1, 1, 1, image.height, image.width),
                    "t_count": 1,
                    "c_count": len(image.getbands()),
                    "z_count": 1,
                    "dtype": "unknown",
                    "pixel_size_um": None,
                }
                metadata["source"] = "tiff-pillow"
                return ThumbnailResult(pixmap=pixmap, metadata=metadata)
        except Exception:
            pass

        if tifffile is None:
            return None

        try:
            with tifffile.TiffFile(str(path)) as tiff:
                if not tiff.series:
                    return None

                series = tiff.series[0]
                level = series.levels[-1] if hasattr(series, "levels") and series.levels else series

                pages = getattr(level, "pages", None)
                if pages is not None and len(pages) > 0:
                    page0 = pages[0]
                    page_shape = tuple(int(v) for v in getattr(page0, "shape", ()) if isinstance(v, (int, np.integer)))
                    plane_pixels = int(np.prod(page_shape)) if page_shape else 0
                    if plane_pixels > self._max_tiff_plane_pixels:
                        metadata = self._load_tiff_metadata(path) or {
                            "shape_tczyx": (1, 1, 1, 1, 1),
                            "t_count": 1,
                            "c_count": 1,
                            "z_count": 1,
                            "dtype": "unknown",
                            "pixel_size_um": None,
                        }
                        metadata["source"] = "tifffile-skip-too-large"
                        metadata["too_large"] = True
                        return ThumbnailResult(pixmap=self._broken_placeholder(size), metadata=metadata)

                    array = page0.asarray()
                else:
                    array = level.asarray(key=0)

            display = self._reduce_tiff_to_display(np.asarray(array), slice_request=slice_request)
            contrasted = self._auto_contrast(display)
            pixmap = self._numpy_to_fitted_qpixmap(contrasted, size)
            metadata = self._load_tiff_metadata(path) or self._metadata_from_tiff_array(np.asarray(array))
            metadata["source"] = "tifffile-fallback"
            return ThumbnailResult(pixmap=pixmap, metadata=metadata)
        except Exception as exc:
            LOGGER.debug("TIFF fallback failed for %s: %s", path, exc)
            return None

    def _load_ims_thumbnail(
        self,
        path: Path,
        size: int,
        *,
        slice_request: dict[str, Any] | None = None,
    ) -> ThumbnailResult | None:
        h5py_mod = self._get_h5py()
        if h5py_mod is None:
            return None

        try:
            with h5py_mod.File(str(path), "r") as ims_file:
                planes: list[np.ndarray] = []
                channel_slice_idx = None
                if slice_request and slice_request.get("mode") == "c":
                    try:
                        channel_slice_idx = int(slice_request.get("index", 0))
                    except Exception:
                        channel_slice_idx = 0
                    channel_slice_idx = max(0, channel_slice_idx)

                dataset_group = ims_file.get("DataSet")
                if dataset_group is not None:
                    resolution_name = self._pick_last_numbered_key(dataset_group.keys(), "ResolutionLevel")
                    if resolution_name is not None:
                        resolution_group = dataset_group[resolution_name]
                        time_name = self._pick_last_numbered_key(resolution_group.keys(), "TimePoint", choose_last=False)
                        if time_name is not None:
                            time_group = resolution_group[time_name]
                            channel_names = self._sorted_numbered_keys(time_group.keys(), "Channel")
                            selected_names = channel_names
                            if channel_slice_idx is not None and channel_names:
                                one = min(channel_slice_idx, len(channel_names) - 1)
                                selected_names = [channel_names[one]]
                            elif channel_slice_idx is None:
                                selected_names = channel_names[:3]

                            for channel_name in selected_names:
                                channel_group = time_group[channel_name]
                                data_ds = channel_group.get("Data")
                                if data_ds is None:
                                    continue
                                plane = self._ims_read_preview_plane(data_ds, slice_request=slice_request)
                                if plane is not None:
                                    planes.append(plane)

                if not planes:
                    discovered = self._discover_ims_data_datasets(ims_file)
                    selected_discovered = discovered
                    if channel_slice_idx is not None and discovered:
                        one = min(channel_slice_idx, len(discovered) - 1)
                        selected_discovered = [discovered[one]]
                    elif channel_slice_idx is None:
                        selected_discovered = discovered[:3]

                    for data_ds in selected_discovered:
                        plane = self._ims_read_preview_plane(data_ds, slice_request=slice_request)
                        if plane is not None:
                            planes.append(plane)

                if not planes:
                    return None

                if len(planes) == 1:
                    display = planes[0]
                else:
                    min_h = min(p.shape[0] for p in planes)
                    min_w = min(p.shape[1] for p in planes)
                    resized = [p[:min_h, :min_w] for p in planes]
                    rgb = np.zeros((min_h, min_w, 3), dtype=np.float32)
                    rgb[:, :, 2] = resized[0]
                    rgb[:, :, 1] = resized[1] if len(resized) > 1 else 0
                    rgb[:, :, 0] = resized[2] if len(resized) > 2 else 0
                    display = rgb

                contrasted = self._auto_contrast(display)
                pixmap = self._numpy_to_fitted_qpixmap(contrasted, size)
                metadata = self._load_ims_metadata(path) or {
                    "shape_tczyx": (1, len(planes), 1, contrasted.shape[0], contrasted.shape[1]),
                    "t_count": 1,
                    "c_count": len(planes),
                    "z_count": 1,
                    "dtype": "unknown",
                    "pixel_size_um": None,
                }
                metadata["source"] = "ims-hdf5"
                return ThumbnailResult(pixmap=pixmap, metadata=metadata)
        except Exception as exc:
            LOGGER.debug("IMS fallback failed for %s: %s", path, exc)
            return None

    def _load_ims_metadata(self, path: Path) -> dict[str, Any] | None:
        h5py_mod = self._get_h5py()
        if h5py_mod is None:
            return None

        try:
            with h5py_mod.File(str(path), "r") as ims_file:
                dataset_group = ims_file.get("DataSet")
                data_ds = None
                time_names: list[str] = []
                channel_names: list[str] = []

                if dataset_group is not None:
                    resolution_name = self._pick_last_numbered_key(dataset_group.keys(), "ResolutionLevel")
                    if resolution_name is not None:
                        resolution_group = dataset_group[resolution_name]
                        time_names = self._sorted_numbered_keys(resolution_group.keys(), "TimePoint")
                        if time_names:
                            first_time = resolution_group[time_names[0]]
                            channel_names = self._sorted_numbered_keys(first_time.keys(), "Channel")
                            if channel_names:
                                data_ds = first_time[channel_names[0]].get("Data")

                discovered = self._discover_ims_data_datasets(ims_file)
                if data_ds is None and discovered:
                    data_ds = discovered[0]

                if data_ds is None:
                    return None

                if not time_names or not channel_names:
                    parsed_time, parsed_channel = self._collect_ims_path_counts(ims_file)
                    if not time_names and parsed_time > 0:
                        time_names = [str(i) for i in range(parsed_time)]
                    if not channel_names and parsed_channel > 0:
                        channel_names = [str(i) for i in range(parsed_channel)]

                shape = tuple(int(v) for v in data_ds.shape)
                if len(shape) >= 3:
                    z_count = int(shape[0])
                    y = int(shape[-2])
                    x = int(shape[-1])
                elif len(shape) == 2:
                    z_count = 1
                    y = int(shape[0])
                    x = int(shape[1])
                else:
                    z_count = 1
                    y = x = 1

                return {
                    "shape_tczyx": (max(1, len(time_names)), max(1, len(channel_names)), z_count, y, x),
                    "t_count": max(1, len(time_names)),
                    "c_count": max(1, len(channel_names)),
                    "z_count": z_count,
                    "dtype": str(data_ds.dtype),
                    "pixel_size_um": None,
                }
        except Exception as exc:
            LOGGER.debug("IMS metadata fallback failed for %s: %s", path, exc)
            return None

    def _discover_ims_data_datasets(self, ims_file) -> list[Any]:
        discovered: list[tuple[str, Any]] = []

        def _visitor(name, obj) -> None:
            if not isinstance(name, str):
                return
            if not name.startswith("DataSet/") or not name.endswith("/Data"):
                return
            if not hasattr(obj, "shape"):
                return
            discovered.append((name, obj))

        try:
            ims_file.visititems(_visitor)
        except Exception:
            return []

        discovered.sort(key=lambda pair: (self._ims_dataset_sort_key(pair[0]), pair[0]))
        return [obj for _, obj in discovered]

    def _ims_dataset_sort_key(self, name: str) -> tuple[int, int, int]:
        level = self._extract_numbered_component(name, "ResolutionLevel")
        time = self._extract_numbered_component(name, "TimePoint")
        channel = self._extract_numbered_component(name, "Channel")
        return (level, time, channel)

    def _extract_numbered_component(self, text: str, prefix: str) -> int:
        match = re.search(rf"{re.escape(prefix)}\s*(\d+)", text)
        if not match:
            return 0
        try:
            return int(match.group(1))
        except Exception:
            return 0

    def _collect_ims_path_counts(self, ims_file) -> tuple[int, int]:
        time_indices: set[int] = set()
        channel_indices: set[int] = set()

        def _visitor(name, obj) -> None:
            if not isinstance(name, str):
                return
            if not name.startswith("DataSet/") or not name.endswith("/Data"):
                return
            if not hasattr(obj, "shape"):
                return

            time_idx = self._extract_numbered_component(name, "TimePoint")
            channel_idx = self._extract_numbered_component(name, "Channel")
            time_indices.add(time_idx)
            channel_indices.add(channel_idx)

        try:
            ims_file.visititems(_visitor)
        except Exception:
            return (0, 0)

        return (len(time_indices), len(channel_indices))

    def _ims_read_preview_plane(
        self,
        dataset,
        *,
        slice_request: dict[str, Any] | None = None,
    ) -> np.ndarray | None:
        shape = tuple(int(v) for v in getattr(dataset, "shape", ()))
        if len(shape) < 2:
            return None

        try:
            if len(shape) >= 3:
                z_len = int(shape[0])
                z_idx = z_len // 2
                if slice_request and slice_request.get("mode") == "z":
                    z_idx = int(slice_request.get("index", z_idx))
                z_idx = max(0, min(z_idx, z_len - 1))
                plane = dataset[z_idx, ...]
            else:
                plane = dataset[...]

            arr = np.asarray(plane, dtype=np.float32)
            if arr.ndim > 2:
                arr = np.squeeze(arr)
                if arr.ndim > 2:
                    arr = arr[0]

            if arr.ndim != 2:
                return None

            pixels = int(arr.shape[0] * arr.shape[1])
            if pixels > self._max_ims_plane_pixels:
                step = int(np.ceil(np.sqrt(pixels / self._max_ims_plane_pixels)))
                step = max(step, 1)
                arr = arr[::step, ::step]

            return arr
        except Exception:
            return None

    def _pick_last_numbered_key(self, keys, prefix: str, *, choose_last: bool = True) -> str | None:
        sorted_keys = self._sorted_numbered_keys(keys, prefix)
        if not sorted_keys:
            return None
        return sorted_keys[-1] if choose_last else sorted_keys[0]

    def _sorted_numbered_keys(self, keys, prefix: str) -> list[str]:
        parsed: list[tuple[int, str]] = []
        for key in keys:
            if not isinstance(key, str):
                continue
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix):].strip()
            try:
                parsed.append((int(suffix), key))
            except Exception:
                continue
        parsed.sort(key=lambda item: item[0])
        return [item[1] for item in parsed]

    def _load_tiff_metadata(self, path: Path) -> dict[str, Any] | None:
        if tifffile is None:
            return None

        try:
            with tifffile.TiffFile(str(path)) as tiff:
                if not tiff.series:
                    return None
                series = tiff.series[0]
                shape = tuple(int(v) for v in getattr(series, "shape", ()) if isinstance(v, (int, np.integer)))
                dtype = getattr(series, "dtype", np.dtype("uint8"))
            return self._metadata_from_tiff_shape(shape, np.dtype(dtype))
        except Exception as exc:
            LOGGER.debug("TIFF metadata fallback failed for %s: %s", path, exc)
            return None

    def _metadata_from_tiff_shape(self, shape: tuple[int, ...], dtype: np.dtype) -> dict[str, Any]:
        t_count = 1
        c_count = 1
        z_count = 1

        if len(shape) == 5:
            t_count, c_count, z_count, y, x = int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]), int(shape[4])
        elif len(shape) == 4:
            t_count, z_count, y, x = int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])
        elif len(shape) == 3:
            if int(shape[-1]) in (3, 4):
                c_count, y, x = int(shape[-1]), int(shape[0]), int(shape[1])
            else:
                z_count, y, x = int(shape[0]), int(shape[1]), int(shape[2])
        elif len(shape) == 2:
            y, x = int(shape[0]), int(shape[1])
        else:
            y = x = 1

        return {
            "shape_tczyx": (t_count, c_count, z_count, y, x),
            "t_count": t_count,
            "c_count": c_count,
            "z_count": z_count,
            "dtype": str(dtype),
            "pixel_size_um": None,
        }

    def _metadata_from_tiff_array(self, arr: np.ndarray) -> dict[str, Any]:
        arr = np.asarray(arr)
        t_count = 1
        c_count = 1
        z_count = 1

        if arr.ndim == 5:
            t_count, c_count, z_count = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
            y, x = int(arr.shape[3]), int(arr.shape[4])
        elif arr.ndim == 4:
            t_count = int(arr.shape[0])
            z_count = int(arr.shape[1])
            y, x = int(arr.shape[2]), int(arr.shape[3])
        elif arr.ndim == 3:
            if arr.shape[-1] in (3, 4):
                c_count = int(arr.shape[-1])
                y, x = int(arr.shape[0]), int(arr.shape[1])
            else:
                z_count = int(arr.shape[0])
                y, x = int(arr.shape[1]), int(arr.shape[2])
        elif arr.ndim == 2:
            y, x = int(arr.shape[0]), int(arr.shape[1])
        else:
            y = x = 1

        return {
            "shape_tczyx": (t_count, c_count, z_count, y, x),
            "t_count": t_count,
            "c_count": c_count,
            "z_count": z_count,
            "dtype": str(arr.dtype),
            "pixel_size_um": None,
        }

    def _reduce_tiff_to_display(
        self,
        arr: np.ndarray,
        *,
        slice_request: dict[str, Any] | None = None,
    ) -> np.ndarray:
        data = np.asarray(arr)

        while data.ndim > 5:
            data = data[0]

        if data.ndim == 5:
            t_idx = 0
            if slice_request and slice_request.get("mode") == "t":
                t_idx = int(slice_request.get("index", 0))
                t_idx = max(0, min(t_idx, data.shape[0] - 1))
            data = data[t_idx]

        if data.ndim == 4:
            z_idx = 0
            if slice_request and slice_request.get("mode") == "z":
                z_idx = int(slice_request.get("index", 0))
                z_idx = max(0, min(z_idx, data.shape[1] - 1))
                data = data[:, z_idx, :, :]
            else:
                data = np.max(data, axis=1)

        if data.ndim == 3:
            if data.shape[-1] in (3, 4):
                return data[..., :3]

            if data.shape[0] == 1:
                return data[0]

            if data.shape[0] in (3, 4):
                rgb = np.zeros((data.shape[1], data.shape[2], 3), dtype=np.float32)
                rgb[:, :, 2] = data[0]
                rgb[:, :, 1] = data[1] if data.shape[0] > 1 else 0
                rgb[:, :, 0] = data[2] if data.shape[0] > 2 else 0
                return rgb

            if slice_request and slice_request.get("mode") == "z":
                z_idx = int(slice_request.get("index", 0))
                z_idx = max(0, min(z_idx, data.shape[0] - 1))
                return data[z_idx]

            return np.max(data, axis=0)

        return np.squeeze(data)

    def _open_bioimage(self, path: Path):
        module = importlib.import_module("bioio")
        bio_image_class = getattr(module, "BioImage")

        name = path.name.lower()
        suffix = path.suffix.lower()

        if name.endswith(".ome.tif") or name.endswith(".ome.tiff"):
            reader_module = importlib.import_module("bioio_ome_tiff")
            reader_class = getattr(reader_module, "Reader")
            return bio_image_class(str(path), reader=reader_class)

        if suffix in self._explicit_bioio_readers:
            reader_module_name = self._explicit_bioio_readers[suffix]
            reader_module = importlib.import_module(reader_module_name)
            reader_class = getattr(reader_module, "Reader")
            return bio_image_class(str(path), reader=reader_class)

        if suffix in {".mvd2", ".acff"}:
            raise RuntimeError("Volocity backend is currently disabled for startup stability")

        raise RuntimeError(f"No explicit BioIO reader configured for extension: {suffix}")

    def _select_lowest_resolution_if_available(self, bio_image) -> None:
        try:
            levels = getattr(bio_image, "resolution_level_dims", None)
            if levels and hasattr(bio_image, "set_resolution_level"):
                lowest = max(0, len(levels) - 1)
                bio_image.set_resolution_level(lowest)
        except Exception:
            LOGGER.debug("Unable to select lowest resolution level", exc_info=True)

    def _collect_bio_metadata(self, bio_image) -> dict[str, Any]:
        dims_obj = getattr(bio_image, "dims", None)
        order = getattr(dims_obj, "order", "TCZYX") or "TCZYX"
        shape = tuple(getattr(dims_obj, "shape", ()))

        dim_map = {d: 1 for d in "TCZYX"}
        if len(order) == len(shape):
            for axis, value in zip(order, shape, strict=False):
                if axis in dim_map:
                    dim_map[axis] = int(value)

        pps = getattr(bio_image, "physical_pixel_sizes", None)
        pixel_um = None
        if pps is not None:
            pixel_um = {"z": getattr(pps, "Z", None), "y": getattr(pps, "Y", None), "x": getattr(pps, "X", None)}

        return {
            "shape_tczyx": (dim_map["T"], dim_map["C"], dim_map["Z"], dim_map["Y"], dim_map["X"]),
            "t_count": dim_map["T"],
            "c_count": dim_map["C"],
            "z_count": dim_map["Z"],
            "dtype": str(getattr(bio_image, "dtype", "unknown")),
            "pixel_size_um": pixel_um,
        }

    def _read_tczyx_data(self, bio_image, *, t_index: int = 0) -> np.ndarray:
        t_index = max(0, int(t_index))
        try:
            arr = bio_image.get_image_dask_data("TCZYX", T=t_index)
            if hasattr(arr, "compute"):
                arr = arr.compute()
            return np.asarray(arr)
        except Exception:
            pass

        try:
            arr = bio_image.get_image_data("TCZYX", T=t_index)
            return np.asarray(arr)
        except Exception:
            pass

        data = getattr(bio_image, "xarray_data", None)
        if data is not None:
            values = getattr(data, "values", data)
            return np.asarray(values)

        fallback = getattr(bio_image, "data", None)
        if fallback is not None:
            return np.asarray(fallback)

        raise RuntimeError("Unable to extract array data from microscopy file")

    def _reduce_tczyx_to_display(
        self,
        data: np.ndarray,
        metadata: dict[str, Any],
        *,
        slice_request: dict[str, Any] | None = None,
    ) -> np.ndarray:
        arr = np.asarray(data)
        while arr.ndim > 5:
            arr = arr[0]

        if arr.ndim == 5:
            arr = arr[0]

        if arr.ndim == 4:
            z_count = int(metadata.get("z_count", arr.shape[1] if arr.shape[1] > 0 else 1))
            if z_count > 1:
                if slice_request and slice_request.get("mode") == "z":
                    z_index = int(slice_request.get("index", 0))
                    z_index = max(0, min(z_index, arr.shape[1] - 1))
                    arr = arr[:, z_index, :, :]
                else:
                    arr = np.max(arr, axis=1)
            else:
                arr = arr[:, 0, :, :]

        if arr.ndim == 3:
            c_count = int(metadata.get("c_count", arr.shape[0]))
            if c_count <= 1 or arr.shape[0] == 1:
                return arr[0]

            if slice_request and slice_request.get("mode") == "c":
                c_index = int(slice_request.get("index", 0))
                c_index = max(0, min(c_index, arr.shape[0] - 1))
                return arr[c_index]

            y, x = arr.shape[1], arr.shape[2]
            rgb = np.zeros((y, x, 3), dtype=np.float32)

            rgb[:, :, 2] = arr[0] if arr.shape[0] > 0 else 0  # C0 -> Blue
            rgb[:, :, 1] = arr[1] if arr.shape[0] > 1 else 0  # C1 -> Green
            rgb[:, :, 0] = arr[2] if arr.shape[0] > 2 else 0  # C2 -> Red
            return rgb

        if arr.ndim == 2:
            return arr

        return np.squeeze(arr)

    def _auto_contrast(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 2:
            return self._auto_contrast_channel(arr)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            channels = [self._auto_contrast_channel(arr[:, :, i]) for i in range(3)]
            return np.stack(channels, axis=-1)
        if arr.ndim == 3 and arr.shape[0] == 3:
            channels = [self._auto_contrast_channel(arr[i, :, :]) for i in range(3)]
            return np.stack(channels, axis=-1)
        return self._auto_contrast_channel(np.squeeze(arr))

    def _auto_contrast_channel(self, channel: np.ndarray) -> np.ndarray:
        data = np.asarray(channel, dtype=np.float32)
        finite = np.isfinite(data)
        if not np.any(finite):
            return np.zeros(data.shape, dtype=np.uint8)

        valid = data[finite]
        non_zero = valid[valid != 0]
        sample = non_zero if non_zero.size > 0 else valid

        lo = float(np.percentile(sample, 1.0))
        hi = float(np.percentile(sample, 99.9))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros(data.shape, dtype=np.uint8)

        clipped = np.clip(data, lo, hi)
        scaled = (clipped - lo) / (hi - lo)
        return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)

    def _extract_exif_thumbnail(self, image: Image.Image) -> Image.Image | None:
        try:
            if "thumbnail" in image.info and isinstance(image.info["thumbnail"], (bytes, bytearray)):
                with Image.open(io.BytesIO(image.info["thumbnail"])) as thumb:
                    return ImageOps.exif_transpose(thumb.convert("RGB"))
        except Exception:
            pass

        try:
            exif = image.getexif()
            raw = exif.get(0x501B) if exif else None
            if isinstance(raw, (bytes, bytearray)):
                with Image.open(io.BytesIO(raw)) as thumb:
                    return ImageOps.exif_transpose(thumb.convert("RGB"))
        except Exception:
            pass

        return None

    def _numpy_to_fitted_qpixmap(self, arr: np.ndarray, size: int) -> QPixmap:
        qimage = self._numpy_to_qimage(arr)
        if qimage.isNull():
            return self._broken_placeholder(size)
        return self._fit_to_square(QPixmap.fromImage(qimage), size)

    def _numpy_to_qimage(self, arr: np.ndarray) -> QImage:
        arr = np.asarray(arr)
        if arr.ndim == 2:
            h, w = arr.shape
            qimage = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
            return qimage.copy()

        if arr.ndim == 3 and arr.shape[-1] == 3:
            h, w, _ = arr.shape
            qimage = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888)
            return qimage.copy()

        return QImage()

    def _pil_to_fitted_qpixmap(self, image: Image.Image, size: int) -> QPixmap:
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")

        if image.mode == "RGB":
            fmt = QImage.Format.Format_RGB888
            raw = image.tobytes("raw", "RGB")
            qimage = QImage(raw, image.width, image.height, image.width * 3, fmt).copy()
        else:
            fmt = QImage.Format.Format_RGBA8888
            raw = image.tobytes("raw", "RGBA")
            qimage = QImage(raw, image.width, image.height, image.width * 4, fmt).copy()

        source = QPixmap.fromImage(qimage)
        return self._fit_to_square(source, size)

    def _fit_to_square(self, source: QPixmap, size: int) -> QPixmap:
        canvas = QPixmap(size, size)
        canvas.fill(QColor(31, 33, 38))

        scaled = source.scaled(
            size,
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        painter = QPainter(canvas)
        x = (size - scaled.width()) // 2
        y = (size - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.end()
        return canvas

    def _broken_placeholder(self, size: int) -> QPixmap:
        canvas = QPixmap(size, size)
        canvas.fill(QColor(53, 26, 26))

        painter = QPainter(canvas)
        painter.setPen(QColor(220, 130, 130))
        painter.drawText(canvas.rect(), Qt.AlignmentFlag.AlignCenter, "Broken\nFile")
        painter.end()
        return canvas
