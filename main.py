import sys
import logging
import os
from pathlib import Path

from PyQt6.QtWidgets import QApplication
import qdarktheme

from ui.main_window import MainWindow


def main() -> int:
    cjdk_cache_dir = Path.home() / ".microscopy_cache" / "cjdk_cache"
    cjdk_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CJDK_CACHE_DIR", str(cjdk_cache_dir))

    logging.getLogger("bioio").setLevel(logging.ERROR)
    logging.getLogger("bioio_base").setLevel(logging.ERROR)
    logging.getLogger("bioio_ome_tiff").setLevel(logging.ERROR)

    app = QApplication(sys.argv)
    if hasattr(qdarktheme, "setup_theme"):
        qdarktheme.setup_theme("dark")
    else:
        app.setStyleSheet(qdarktheme.load_stylesheet("dark"))

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
