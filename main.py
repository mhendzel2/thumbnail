import sys
import logging

from PyQt6.QtWidgets import QApplication
import qdarktheme

from ui.main_window import MainWindow


def main() -> int:
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
