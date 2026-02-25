from PyQt6.QtCore import QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFontMetrics, QPainter, QPen, QFileSystemModel
from PyQt6.QtCore import QSortFilterProxyModel
from PyQt6.QtWidgets import QStyle, QStyleOptionViewItem, QStyledItemDelegate


class ThumbnailDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._thumb_size = QSize(160, 160)
        self._item_size = QSize(180, 210)
        self._thumbnail_provider = None
        self._thumbnail_requester = None
        self._metadata_provider = None

    def set_thumbnail_provider(self, provider):
        self._thumbnail_provider = provider

    def set_thumbnail_requester(self, requester):
        self._thumbnail_requester = requester

    def set_metadata_provider(self, provider):
        self._metadata_provider = provider

    def _file_path_for_index(self, index) -> str:
        model = index.model()
        if isinstance(model, QFileSystemModel):
            return model.filePath(index)
        if isinstance(model, QSortFilterProxyModel):
            source_index = model.mapToSource(index)
            source_model = model.sourceModel()
            if isinstance(source_model, QFileSystemModel) and source_index.isValid():
                return source_model.filePath(source_index)
        return ""

    def _is_dir_for_index(self, index) -> bool:
        model = index.model()
        if isinstance(model, QFileSystemModel):
            return bool(model.isDir(index))
        if isinstance(model, QSortFilterProxyModel):
            source_index = model.mapToSource(index)
            source_model = model.sourceModel()
            if isinstance(source_model, QFileSystemModel) and source_index.isValid():
                return bool(source_model.isDir(source_index))
        return False

    def sizeHint(self, option: QStyleOptionViewItem, index):
        return self._item_size

    def paint(self, painter, option: QStyleOptionViewItem, index):
        if painter is None:
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = option.rect.adjusted(8, 8, -8, -8)
        is_dir = self._is_dir_for_index(index)

        if option.state & QStyle.StateFlag.State_Selected:
            painter.setPen(QPen(QColor(80, 160, 255, 220), 2))
        else:
            painter.setPen(QPen(QColor(90, 90, 90, 180), 1))

        if is_dir:
            painter.setBrush(QColor(56, 54, 42, 220))
        else:
            painter.setBrush(QColor(44, 48, 56, 220))

        thumb_rect = QRectF(
            rect.left(),
            rect.top(),
            self._thumb_size.width(),
            self._thumb_size.height(),
        )
        painter.drawRoundedRect(thumb_rect, 10, 10)

        file_path = self._file_path_for_index(index)

        if is_dir:
            folder_icon = index.data(Qt.ItemDataRole.DecorationRole)
            if folder_icon is None and option.widget is not None:
                style = option.widget.style()
                if style is not None:
                    folder_icon = style.standardIcon(QStyle.StandardPixmap.SP_DirIcon)

            if folder_icon is not None:
                target = thumb_rect.adjusted(18, 18, -18, -18).toRect()
                icon_pixmap = folder_icon.pixmap(target.size())
                x = target.x() + (target.width() - icon_pixmap.width()) // 2
                y = target.y() + (target.height() - icon_pixmap.height()) // 2
                painter.drawPixmap(x, y, icon_pixmap)
        else:
            pixmap = self._thumbnail_provider(file_path) if self._thumbnail_provider and file_path else None
            if pixmap is not None and not pixmap.isNull():
                target = thumb_rect.adjusted(2, 2, -2, -2).toRect()
                painter.drawPixmap(target, pixmap)
            elif self._thumbnail_requester and file_path:
                self._thumbnail_requester(file_path, index)

        metadata = self._metadata_provider(file_path) if self._metadata_provider and file_path else {}
        self._paint_badges(painter, thumb_rect, metadata)

        name = index.data(Qt.ItemDataRole.DisplayRole) or ""
        text_rect = QRectF(
            rect.left(),
            thumb_rect.bottom() + 8,
            self._thumb_size.width(),
            rect.bottom() - thumb_rect.bottom() - 8,
        )

        metrics = QFontMetrics(painter.font())
        elided = metrics.elidedText(name, Qt.TextElideMode.ElideRight, int(text_rect.width()))
        painter.setPen(QColor(220, 220, 220))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter, elided)

        painter.restore()

    def _paint_badges(self, painter: QPainter, thumb_rect: QRectF, metadata: dict) -> None:
        badges: list[str] = []
        z_count = int(metadata.get("z_count", 1) or 1)
        t_count = int(metadata.get("t_count", 1) or 1)
        c_count = int(metadata.get("c_count", 1) or 1)

        if z_count > 1:
            badges.append(f"[Z: {z_count}]")
        if t_count > 1:
            badges.append(f"[T: {t_count}]")
        if c_count > 1:
            badges.append(f"[C: {c_count}]")

        if not badges:
            return

        metrics = QFontMetrics(painter.font())
        x = int(thumb_rect.right()) - 6
        y = int(thumb_rect.top()) + 6

        for badge in badges:
            width = metrics.horizontalAdvance(badge) + 14
            height = metrics.height() + 6
            x0 = x - width
            rect = QRectF(x0, y, width, height)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(22, 24, 30, 180))
            painter.drawRoundedRect(rect, 8, 8)

            painter.setPen(QColor(220, 230, 255, 230))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, badge)
            y += height + 4
