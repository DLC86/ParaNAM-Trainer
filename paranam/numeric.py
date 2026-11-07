"""Decimal entry accepts either separator, including paste, in every locale."""

from PySide6.QtWidgets import QDoubleSpinBox


class DecimalSpinBox(QDoubleSpinBox):
    def _decimal_text(self, text):
        return text.replace(".", self.locale().decimalPoint()).replace(
            ",", self.locale().decimalPoint()
        )

    def validate(self, text, position):
        return super().validate(self._decimal_text(text), position)

    def valueFromText(self, text):
        return super().valueFromText(self._decimal_text(text))

    def fixup(self, text):
        return super().fixup(self._decimal_text(text))
