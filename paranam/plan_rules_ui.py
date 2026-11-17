"""Capture Plan Avoid/Mute rule editor with independent control conditions."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QGridLayout,
    QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from nam.capture.params import SwitchSpec
from nam.capture.plan_rules import PlanRule, validate_plan_rules
from .numeric import DecimalSpinBox


def describe_rule(rule):
    terms = []
    for name in rule.controls:
        operator, value = rule.condition_for(name)
        text = value if isinstance(value, str) else f"{value:g}"
        terms.append(f"{name} {operator} {text}")
    return ("Mute when " if rule.kind == "mute" else "Avoid when ") + " AND ".join(terms)


class PlanRuleDialog(QDialog):
    def __init__(self, parent, controls, rule=None):
        super().__init__(parent)
        self.setWindowTitle("Edit plan rule" if rule else "Add plan rule")
        self.resize(590, 400)
        self.controls = {control.name: control for control in controls}
        self.rule = None
        self.rows = {}
        layout = QVBoxLayout(self)
        detail = QLabel(
            "All selected conditions must be true (AND). Avoid excludes unstable "
            "combinations. Mute uses ordinary full-length synthetic silent captures "
            "at selected positions. Each control has its own condition."
        )
        detail.setWordWrap(True)
        layout.addWidget(detail)
        self.rule_kind = QComboBox()
        self.rule_kind.setAccessibleName("Rule type")
        self.rule_kind.addItem("Avoid — unstable / do not capture", "avoid")
        self.rule_kind.addItem("Mute — train silence without recording", "mute")
        self.rule_kind.setCurrentIndex(
            max(0, self.rule_kind.findData(rule.kind if rule else "avoid"))
        )
        layout.addWidget(self.rule_kind)

        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setMinimumHeight(180)
        area.setMaximumHeight(430)
        content = QWidget()
        grid = QGridLayout(content)
        for col, heading in enumerate(("Use", "Control", "Condition", "Value")):
            grid.addWidget(QLabel(heading), 0, col)
        for i, control in enumerate(controls, 1):
            checked = QCheckBox()
            checked.setAccessibleName(f"Include {control.name} in rule")
            checked.setChecked(rule is not None and control.name in rule.controls)
            grid.addWidget(checked, i, 0)
            grid.addWidget(QLabel(control.name), i, 1)
            operator = QComboBox()
            operator.setAccessibleName(f"Operator for {control.name}")
            is_switch = isinstance(control, SwitchSpec)
            operator.addItems(["="] if is_switch else ["<", "≤", "=", "≥", ">"])
            grid.addWidget(operator, i, 2)
            if is_switch:
                value = QComboBox()
                value.addItems(list(control.enum_names))
            else:
                value = DecimalSpinBox()
                value.setDecimals(6)
                value.setRange(-10000, 10000)
                value.setValue(control.default)
            value.setAccessibleName(f"Threshold for {control.name}")
            grid.addWidget(value, i, 3)
            if rule is not None and control.name in rule.controls:
                op, threshold = rule.condition_for(control.name)
                operator.setCurrentText(op)
                if is_switch:
                    value.setCurrentText(str(threshold))
                else:
                    value.setValue(float(threshold))
            else:
                operator.setCurrentText("=" if is_switch else "≥")
            checked.toggled.connect(self.update_fields)
            self.rows[control.name] = (checked, operator, value)
        grid.setColumnStretch(3, 1)
        area.setWidget(content)
        layout.addWidget(area)

        select_all = QPushButton("Select all")
        select_all.clicked.connect(self.select_all)
        layout.addWidget(select_all)
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.update_fields()

    def selected(self):
        return [
            name for name, (checked, _, _) in self.rows.items()
            if checked.isChecked()
        ]

    def select_all(self):
        for checked, _, _ in self.rows.values():
            checked.setChecked(True)
        self.update_fields()

    def update_fields(self, *_):
        selected = self.selected()
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(selected))
        for checked, operator, value in self.rows.values():
            operator.setEnabled(checked.isChecked())
            value.setEnabled(checked.isChecked())

    def accept(self):
        try:
            selected = self.selected()
            if not selected:
                raise ValueError("Select at least one control")
            conditions = {}
            for name in selected:
                _, operator, field = self.rows[name]
                control = self.controls[name]
                threshold = (
                    field.currentText() if isinstance(control, SwitchSpec)
                    else field.value()
                )
                conditions[name] = (operator.currentText(), threshold)
            first_op, first_val = conditions[selected[0]]
            rule = PlanRule(
                controls=tuple(selected), operator=first_op, value=first_val,
                kind=self.rule_kind.currentData(),
                operators={
                    name: op for name, (op, _) in conditions.items()
                    if op != first_op
                },
                thresholds={
                    name: value for name, (_, value) in conditions.items()
                    if value != first_val
                },
            )
            validate_plan_rules(
                [c.to_param_spec() for c in self.controls.values()], [rule]
            )
        except ValueError as error:
            self.error.setText(str(error))
            return
        self.rule = rule
        super().accept()
