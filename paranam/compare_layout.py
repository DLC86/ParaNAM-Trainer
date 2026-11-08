"""Compact controls and visible asset lists for Compare."""

from PySide6.QtCore import QEvent, QRect, QSize, Qt, QTimer, QItemSelectionModel
from PySide6.QtWidgets import QFrame, QLayout, QListWidget, QAbstractItemView


class ControlPanel(QFrame):
    """Give the surrounding scroll area the height of every wrapped control row."""

    def fit_rows(self):
        if self.layout() is not None:
            height = (
                self.layout().heightForWidth(self.contentsRect().width())
                + 2 * self.frameWidth()
            )
            if height != self.minimumHeight() or height != self.maximumHeight():
                self.setFixedHeight(height)
                self.updateGeometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_rows()

    def event(self, event):
        result = super().event(event)
        if event.type() == QEvent.Type.LayoutRequest:
            self.fit_rows()
        return result


class ControlFlowLayout(QLayout):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = []
        self.setSpacing(24)

    def setHorizontalSpacing(self, spacing):
        self.setSpacing(spacing)

    def addItem(self, item):
        self.items.append(item)
        self.invalidate()

    def count(self):
        return len(self.items)

    def itemAt(self, index):
        return self.items[index] if 0 <= index < len(self.items) else None

    def takeAt(self, index):
        if 0 <= index < len(self.items):
            item = self.items.pop(index)
            self.invalidate()
            return item
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QRect(0, 0, width, 0), False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        result = QSize()
        for item in self.items:
            widget = item.widget()
            result = result.expandedTo(
                widget.minimumSize() if widget else item.minimumSize()
            )
        margins = self.contentsMargins()
        return result + QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )

    def _arrange(self, rect, apply):
        margins = self.contentsMargins()
        area = rect.adjusted(
            margins.left(), margins.top(), -margins.right(), -margins.bottom()
        )
        x, y, height = area.x(), area.y(), 0
        for item in self.items:
            widget = item.widget()
            size = (
                widget.sizeHint().expandedTo(widget.minimumSize())
                if widget
                else item.sizeHint().expandedTo(item.minimumSize())
            )
            if x > area.x() and x + size.width() > area.right() + 1:
                x, y, height = area.x(), y + height + self.spacing(), 0
            if apply:
                item.setGeometry(QRect(x, y, size.width(), size.height()))
            x += size.width() + self.spacing()
            height = max(height, size.height())
        return y + height - rect.y() + margins.bottom()


class AssetList(QListWidget):
    """A list backed by the existing selection model, with one load per selection."""

    def __init__(self, selector, parent=None):
        super().__init__(parent)
        self.selector = selector
        self.pending = False
        self.setMinimumHeight(180)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        selector.installEventFilter(self)
        selector.currentIndexChanged.connect(self.sync)
        for signal in (
            selector.model().rowsInserted,
            selector.model().rowsRemoved,
            selector.model().modelReset,
            selector.model().dataChanged,
        ):
            signal.connect(self.schedule_sync)
        self.currentRowChanged.connect(selector.setCurrentIndex)
        self.sync()

    def schedule_sync(self, *_):
        if not self.pending:
            self.pending = True
            QTimer.singleShot(0, self.sync)

    def sync(self, *_):
        self.pending = False
        if self.count() == self.selector.count() and all(
            self.item(i).text() == self.selector.itemText(i)
            and self.item(i).data(Qt.ItemDataRole.UserRole) == self.selector.itemData(i)
            for i in range(self.count())
        ):
            self.blockSignals(True)
            self.setCurrentRow(
                self.selector.currentIndex(), QItemSelectionModel.SelectionFlag.NoUpdate
            )
            self.blockSignals(False)
            self.setEnabled(self.selector.isEnabled())
            return
        self.blockSignals(True)
        self.clear()
        for index in range(self.selector.count()):
            self.addItem(self.selector.itemText(index))
            self.item(index).setData(
                Qt.ItemDataRole.UserRole, self.selector.itemData(index)
            )
            self.item(index).setToolTip(
                self.selector.itemData(index, Qt.ItemDataRole.ToolTipRole)
                or self.selector.itemText(index)
            )
        self.setCurrentRow(self.selector.currentIndex())
        self.blockSignals(False)
        self.setEnabled(self.selector.isEnabled())

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.EnabledChange:
            self.setEnabled(self.selector.isEnabled())
        return False

