"""Table selection must not replace per-row status foreground colors."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QStyleOptionViewItem, QTableWidget, QTableWidgetItem

from paranam.ui_widgets import PreserveSelectionTextDelegate


def test_selected_table_cell_uses_its_original_status_foreground(qapp):
    table = QTableWidget(1, 1)
    delegate = PreserveSelectionTextDelegate(table)
    table.setItemDelegate(delegate)
    item = QTableWidgetItem("Completed")
    item.setForeground(QColor("#00ffff"))
    table.setItem(0, 0, item)
    table.selectRow(0)

    option = QStyleOptionViewItem()
    delegate.initStyleOption(option, table.model().index(0, 0))
    for group in (
        QPalette.ColorGroup.Active,
        QPalette.ColorGroup.Inactive,
        QPalette.ColorGroup.Disabled,
    ):
        assert option.palette.color(group, QPalette.ColorRole.HighlightedText) == QColor(
            "#00ffff"
        )


def test_selected_cell_default_foreground_remains_unchanged(qapp):
    table = QTableWidget(1, 1)
    delegate = PreserveSelectionTextDelegate(table)
    table.setItemDelegate(delegate)
    table.setItem(0, 0, QTableWidgetItem("Pending"))
    option = QStyleOptionViewItem()
    expected = option.palette.brush(QPalette.ColorRole.HighlightedText)
    delegate.initStyleOption(option, table.model().index(0, 0))
    assert table.item(0, 0).data(Qt.ItemDataRole.ForegroundRole) is None
    assert option.palette.brush(QPalette.ColorRole.HighlightedText) == expected
