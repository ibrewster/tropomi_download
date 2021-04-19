"""GradientScale.py

Qt Widget that can display a color bar defined by a QLinearGradient
"""
from PySide2.QtWidgets import QLabel
from PySide2.QtGui import (QPainter,
                           QLinearGradient,
                           QGradient,
                           QMatrix)

from PySide2.QtCore import (QRectF,
                            Qt)


class GradientWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.gradient = QLinearGradient(0, 0, 0, self.height())
        self.labels = {}
        self.margin = 10
        self._barThickness = 20
        self._labelMargin = 2

        self._maxLabelWidth = None
        self._maxLabelHeight = None

        self._orientation = 'Vertical'
        self.setStyleSheet("background-color: rgba(255,0,255,0)")
        self.setAttribute(Qt.WA_NoSystemBackground)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.HighQualityAntialiasing)

        # determine max width/height of all labels
        if self._maxLabelWidth is None or self._maxLabelHeight is None:
            self._maxLabelWidth = 0
            self._maxLabelHeight = 0
            for k in self.labels.values():
                b = painter.boundingRect(QRectF(0, 0, 0, 0), Qt.AlignLeft | Qt.AlignVCenter, str(k))
                self._maxLabelWidth = max(self._maxLabelWidth, b.width())
                self._maxLabelHeight = max(self._maxLabelHeight, b.height())

        barRect = self.rect()

        if self._orientation == 'Vertical':
            # translate Y axis
            matrix = QMatrix()
            matrix.translate(0, self.height() - self.margin)
            matrix.scale(1, -1)
            painter.setMatrix(matrix)

            self.gradient.setFinalStop(0, self.height() - self.margin)
            barRect.setWidth(self._barThickness)
            barRect.setHeight(barRect.height() - 2 * self.margin)
            painter.fillRect(barRect, self.gradient)

            # restore normal coordinates
            painter.scale(1, -1)

            self.setMinimumWidth(self._barThickness + self._labelMargin + self._maxLabelWidth)

        elif self._orientation == 'Horizontal':
            self.margin = self._maxLabelWidth / 2 + self._labelMargin
            barRect.setHeight(self._barThickness)
            barRect.setLeft(self.margin)  # reduces width by margin (as opposed to shifting)

            # Reduce the width by another margin to pull off the right hand side
            barRect.setWidth(barRect.width() - self.margin)

            painter.fillRect(barRect, self.gradient)

            line_step = barRect.width() / 20
            pos = barRect.left()
            while pos <= barRect.right():
                painter.drawLine(pos, 2 * (barRect.bottom() - barRect.top()) / 3,
                                 pos, barRect.bottom())
                pos += line_step

            self.setMinimumHeight(self._maxLabelHeight + self._barThickness + self._labelMargin)

        for pos, label in self.labels.items():
            # Figure coordinate position. 1=height-margin for vertical, or width-margin for horizontal
            if self._orientation == 'Vertical':
                lpos = -1 * ((self.height() - (2 * self.margin)) * pos)
                painter.drawText(self._barThickness + self._labelMargin, (.5 * self._maxLabelHeight) + lpos, label)
            elif self._orientation == 'Horizontal':
                text_rect = painter.boundingRect(QRectF(0, 0, 0, 0), Qt.AlignLeft, str(label))
                lpos = ((self.width() - (2 * self.margin)) * pos)  # Center position
                lleft = lpos - text_rect.width() / 2
                painter.drawText(lleft + self.margin, self.height() - 1, label)

    def setGradient(self, g):
        self.gradient = g
        self.gradient.setCoordinateMode(QGradient.ObjectMode)

        # Make sure we go end-to-end
        self.gradient.setFinalStop(1, 0)
        self.gradient.setStart(0, 0)

        self.repaint()

    def setLabels(self, l):
        self.labels = l
        # recalculate self.labelWidth on next paint
        self._maxLabelWidth = None
        self._maxLabelHeight = None
        self.repaint()

    def setOrientation(self, orientation):
        if not orientation in ('Horizontal', 'Vertical'):
            raise TypeError("Orientation must be one of 'Horizontal' or 'Vertical'")

        self._orientation = orientation
        self.repaint()
