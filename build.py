from __future__ import annotations

from pathlib import Path

import PyInstaller.__main__


def build() -> None:
    root = Path(__file__).resolve().parent
    entry = root / "main.py"

    PyInstaller.__main__.run(
        [
            str(entry),
            "--name",
            "MicroscopyThumbnailViewer",
            "--noconfirm",
            "--windowed",
            "--clean",
            "--hidden-import=bioio",
            "--hidden-import=bioio_bioformats",
            "--hidden-import=bioio_czi",
            "--hidden-import=bioio_nd2",
            "--hidden-import=bioio_ome_tiff",
        ]
    )


if __name__ == "__main__":
    build()
