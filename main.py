import sys

from PyQt6.QtWidgets import QApplication
import qdarktheme

from ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
