"""Editor for a conjunction of capture-plan exclusions."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QStackedWidget, QVBoxLayout,
)

from nam.capture.params import SwitchSpec
from nam.capture.plan_rules import PlanRule, validate_plan_rules
from .numeric import DecimalSpinBox


def describe_rule(rule):
    value = rule.value if isinstance(rule.value, str) else f"{rule.value:g}"
    return f"Avoid all {rule.operator} {value}  ·  {', '.join(rule.controls)}"


class PlanRuleDialog(QDialog):
    def __init__(self, parent, controls, rule=None):
        super().__init__(parent)
        self.setWindowTitle("Edit plan rule" if rule else "Add plan rule")
        self.controls = {control.name: control for control in controls}
        self.rule = None
        layout = QVBoxLayout(self)
        detail = QLabel("Skip a setting when every selected control matches the condition.")
        detail.setWordWrap(True)
        layout.addWidget(detail)
        self.selection = QListWidget()
        self.selection.setAccessibleName("Controls in this rule")
        for name in self.controls:
            item = QListWidgetItem(name, self.selection)
            item.setCheckState(Qt.CheckState.Checked if rule and name in rule.controls
                               else Qt.CheckState.Unchecked)
        layout.addWidget(self.selection)
        select_all = QPushButton("Select all")
        select_all.clicked.connect(self.select_all)
        layout.addWidget(select_all)

        form = QFormLayout()
        self.operator = QComboBox()
        self.operator.addItems(["<", ">", "="])
        form.addRow("Avoid all", self.operator)
        self.number = DecimalSpinBox()
        self.number.setRange(-10000, 10000)
        self.number.setDecimals(6)
        self.position = QComboBox()
        self.value = QStackedWidget()
        self.value.addWidget(self.number)
        self.value.addWidget(self.position)
        form.addRow("Value", self.value)
        layout.addLayout(form)
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                        | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.selection.itemChanged.connect(self.update_fields)
        self.update_fields()
        if rule:
            self.operator.setCurrentText(rule.operator)
            if isinstance(rule.value, str):
                self.position.setCurrentText(rule.value)
            else:
                self.number.setValue(rule.value)

    def selected(self):
        return [
            item.text() for item in (self.selection.item(i) for i in range(self.selection.count()))
            if item.checkState() == Qt.CheckState.Checked
        ]

    def select_all(self):
        self.selection.blockSignals(True)
        for i in range(self.selection.count()):
            self.selection.item(i).setCheckState(Qt.CheckState.Checked)
        self.selection.blockSignals(False)
        self.update_fields()

    def update_fields(self):
        selected = [self.controls[name] for name in self.selected()]
        switches = [c for c in selected if isinstance(c, SwitchSpec)]
        self.operator.setEnabled(not switches)
        self.value.setCurrentIndex(int(bool(switches)))
        message = ""
        if switches:
            self.operator.setCurrentText("=")
            previous = self.position.currentText()
            shared = [name for name in switches[0].enum_names
                      if all(name in c.enum_names for c in switches)]
            self.position.clear()
            self.position.addItems(shared)
            self.position.setCurrentText(previous)
            if len(switches) != len(selected):
                message = "Choose either knobs or switches for this condition."
            elif not shared:
                message = "The selected switches need a shared position name."
        self.error.setText(message)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(selected) and not message)

    def accept(self):
        try:
            names = self.selected()
            rule = PlanRule(
                controls=names,
                operator=self.operator.currentText(),
                value=self.position.currentText() if self.value.currentIndex() else self.number.value(),
            )
            validate_plan_rules([c.to_param_spec() for c in self.controls.values()], [rule])
        except ValueError as error:
            self.error.setText(str(error))
            return
        self.rule = rule
        super().accept()
