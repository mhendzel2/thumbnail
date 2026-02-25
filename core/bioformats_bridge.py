from __future__ import annotations

import base64
import io
import json
import sys

import numpy as np
from PIL import Image

from bioio import BioImage
from bioio_bioformats import Reader


def _normalize(channel: np.ndarray) -> np.ndarray:
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


def _to_display(arr: np.ndarray, mode: str, index: int) -> np.ndarray:
    data = np.asarray(arr)

    while data.ndim > 5:
        data = data[0]

    if data.ndim == 5:
        data = data[0]

    if data.ndim == 4:
        if mode == "z" and data.shape[1] > 0:
            z_idx = max(0, min(int(index), data.shape[1] - 1))
            data = data[:, z_idx, :, :]
        else:
            data = np.max(data, axis=1) if data.shape[1] > 1 else data[:, 0, :, :]

    if data.ndim == 3:
        if mode == "c" and data.shape[0] > 0:
            c_idx = max(0, min(int(index), data.shape[0] - 1))
            return data[c_idx]

        if data.shape[0] == 1:
            return data[0]

        if data.shape[0] in (3, 4):
            rgb = np.zeros((data.shape[1], data.shape[2], 3), dtype=np.float32)
            rgb[:, :, 2] = data[0]
            rgb[:, :, 1] = data[1] if data.shape[0] > 1 else 0
            rgb[:, :, 0] = data[2] if data.shape[0] > 2 else 0
            return rgb

        return np.max(data, axis=0)

    if data.ndim == 2:
        return data

    return np.squeeze(data)


def main() -> int:
    path = sys.argv[1]
    size = int(sys.argv[2])
    mode = str(sys.argv[3] or "").lower()
    index = int(sys.argv[4])
    metadata_only = bool(int(sys.argv[5]))

    img = BioImage(path, reader=Reader)

    dims = getattr(img, "dims", None)
    order = getattr(dims, "order", "TCZYX") if dims is not None else "TCZYX"
    shape = tuple(getattr(dims, "shape", ())) if dims is not None else ()
    dim_map = {d: 1 for d in "TCZYX"}
    if len(order) == len(shape):
        for axis, value in zip(order, shape, strict=False):
            if axis in dim_map:
                dim_map[axis] = int(value)

    response: dict[str, object] = {
        "ok": True,
        "shape_tczyx": [dim_map["T"], dim_map["C"], dim_map["Z"], dim_map["Y"], dim_map["X"]],
        "t_count": dim_map["T"],
        "c_count": dim_map["C"],
        "z_count": dim_map["Z"],
        "dtype": str(getattr(img, "dtype", "unknown")),
    }

    if metadata_only:
        print(json.dumps(response))
        return 0

    arr = img.get_image_dask_data("TCZYX", T=0)
    if hasattr(arr, "compute"):
        arr = arr.compute()
    arr = np.asarray(arr)

    display = np.asarray(_to_display(arr, mode, index))

    if display.ndim == 2:
        out = _normalize(display)
        image = Image.fromarray(out, mode="L")
    else:
        channels = [_normalize(display[:, :, i]) for i in range(3)]
        out = np.stack(channels, axis=-1)
        image = Image.fromarray(out, mode="RGB")

    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    response["png_b64"] = base64.b64encode(buffer.getvalue()).decode("ascii")

    print(json.dumps(response))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        raise SystemExit(1)
