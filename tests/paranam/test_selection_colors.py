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


def test_selected_running_capture_row_renders_green_in_every_column(qapp):
    """The selected row must not turn white while only Status stays green."""
    table = QTableWidget(1, 3)
    table.setStyleSheet(
        "QTableWidget { background: #171717; color: #f0f0f0; } "
        "QTableWidget::item:selected { background: #3f3f3f; }"
    )
    table.setItemDelegate(PreserveSelectionTextDelegate(table))
    table.horizontalHeader().hide()
    table.verticalHeader().hide()
    table.setShowGrid(False)
    table.resize(620, 120)
    for column, value in enumerate(("Train", "Gain=10", "Running")):
        table.setColumnWidth(column, 195)
        item = QTableWidgetItem(value)
        item.setForeground(QColor("#00ff00"))
        table.setItem(0, column, item)
    table.setRowHeight(0, 45)
    table.selectRow(0)
    table.show()
    qapp.processEvents()

    image = table.viewport().grab().toImage()
    for column in range(3):
        rect = table.visualItemRect(table.item(0, column))
        green_pixels = 0
        for y in range(rect.top() + 2, rect.bottom() - 1):
            for x in range(rect.left() + 2, rect.right() - 1):
                pixel = image.pixelColor(x, y)
                if pixel.green() > 175 and pixel.red() < 110 and pixel.blue() < 110:
                    green_pixels += 1
        assert green_pixels > 5, f"Selected column {column} lost its green text"

        background = image.pixelColor(rect.right() - 8, rect.center().y())
        assert background == QColor("#3f3f3f"), (
            f"Selected column {column} lost its selection background: {background.name()}"
        )
    table.close()


def test_selected_capture_row_preserves_warning_and_completed_text(qapp):
    """Status colors must survive actual stylesheet rendering, not only palette setup."""
    table = QTableWidget(1, 2)
    table.setStyleSheet(
        "QTableWidget { background: #171717; } "
        "QTableWidget::item:selected { background: #3f3f3f; }"
    )
    table.setItemDelegate(PreserveSelectionTextDelegate(table))
    table.horizontalHeader().hide()
    table.verticalHeader().hide()
    table.resize(450, 110)
    for column, (value, color) in enumerate(
        (("Captured", "#00ffff"), ("Warning", "#ffff00"))
    ):
        table.setColumnWidth(column, 200)
        item = QTableWidgetItem(value)
        item.setForeground(QColor(color))
        table.setItem(0, column, item)
    table.selectRow(0)
    table.show()
    qapp.processEvents()
    image = table.viewport().grab().toImage()

    for column in range(2):
        rect = table.visualItemRect(table.item(0, column))
        pixels = [
            image.pixelColor(x, y)
            for y in range(rect.top() + 2, rect.bottom() - 1)
            for x in range(rect.left() + 2, rect.right() - 1)
        ]
        if column == 0:
            assert any(p.green() > 170 and p.blue() > 170 and p.red() < 110 for p in pixels)
        else:
            assert any(p.red() > 170 and p.green() > 170 and p.blue() < 110 for p in pixels)
    table.close()


def test_plan_lhs_selection_is_saved_and_detected_as_a_plan_change(window, workspace):
    from paranam.workspace import Workspace

    window._attach(workspace)
    assert window.lhs_method.currentData() == "standard"
    window.lhs_method.setCurrentIndex(window.lhs_method.findData("maximin"))
    assert workspace.preferences["lhs_method"] == "maximin"
    assert workspace.plan_changed()
    assert Workspace(workspace.directory).preferences["lhs_method"] == "maximin"
