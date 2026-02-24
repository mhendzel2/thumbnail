from __future__ import annotations

import io
import importlib
import logging
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

LOGGER = logging.getLogger(__name__)

STANDARD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MICROSCOPY_EXTENSIONS = {".czi", ".nd2", ".tiff", ".tif", ".ome.tif", ".ome.tiff"}


@dataclass(slots=True)
class ThumbnailResult:
    pixmap: QPixmap
    metadata: dict[str, Any]


class ImageEngine:
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
        return name.endswith(".tif") or name.endswith(".tiff")

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
        if tifffile is None:
            return None

        try:
            with tifffile.TiffFile(str(path)) as tiff:
                if not tiff.series:
                    return None

                series = tiff.series[0]
                if hasattr(series, "levels") and series.levels:
                    array = series.levels[-1].asarray()
                else:
                    array = series.asarray()

            display = self._reduce_tiff_to_display(np.asarray(array), slice_request=slice_request)
            contrasted = self._auto_contrast(display)
            pixmap = self._numpy_to_fitted_qpixmap(contrasted, size)
            metadata = self._metadata_from_tiff_array(np.asarray(array))
            metadata["source"] = "tifffile-fallback"
            return ThumbnailResult(pixmap=pixmap, metadata=metadata)
        except Exception as exc:
            LOGGER.warning("TIFF fallback failed for %s: %s", path, exc)
            return None

    def _load_tiff_metadata(self, path: Path) -> dict[str, Any] | None:
        if tifffile is None:
            return None

        try:
            with tifffile.TiffFile(str(path)) as tiff:
                if not tiff.series:
                    return None
                array = tiff.series[0].asarray()
            return self._metadata_from_tiff_array(np.asarray(array))
        except Exception as exc:
            LOGGER.warning("TIFF metadata fallback failed for %s: %s", path, exc)
            return None

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
        return bio_image_class(str(path))

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
