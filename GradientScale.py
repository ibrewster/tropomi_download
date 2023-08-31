from PySide6.QtWidgets import QLabel
from PySide6.QtGui import (QPainter,
                           QLinearGradient,
                           QTransform,
                           QTextDocument,
                           QFontMetrics)

from PySide6.QtCore import QPointF


class GradientWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.gradient = QLinearGradient(0, 0, 0, self.height())
        self.labels = {}
        self.margin_bottom = 40
        self.margin_top = 5
        self._barThickness = 20
        self._labelMargin = 2
        self._title = None
        self._title_height = 0
        self._title_width = 0

        self._maxLabelWidth = None
        self._maxLabelHeight = None

        self._orientation = 'Vertical'

    def get_gradient_rect(self):
        return self._barRect

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setOpacity(0.9)
        painter.setRenderHint(QPainter.Antialiasing)

        self._barRect = self.rect()

        if self._orientation == 'Vertical':
            title_transform = QTransform()
            title_transform.rotate(-90)
            title_transform.translate(-1 * (self.height() / 2) - .5 * self._title_width,
                                      self.width() - self._title_height - self._labelMargin)

            # translate Y axis
            transform = QTransform()
            transform.translate(0, self.height() - self.margin_bottom)
            transform.scale(1, -1)
            painter.setTransform(transform)

            self.gradient.setFinalStop(0, self.height() - self.margin_bottom - self.margin_top)
            self._barRect.setWidth(self._barThickness)
            self._barRect.setHeight(self._barRect.height() - self.margin_bottom - self.margin_top)
            painter.fillRect(self._barRect, self.gradient)

            # restore normal coordinates
            painter.scale(1, -1)

            line_step = self._barRect.height() / 20
            pos = self._barRect.top() + 1
            while pos <= self._barRect.bottom() + line_step:
                draw_pos = pos + self.margin_bottom + self.margin_top - self.height()
                painter.drawLine(.666 * self._barRect.width(), draw_pos,
                                 self._barRect.right(), draw_pos)
                pos += line_step

        elif self._orientation == 'Horizontal':
            title_transform = QTransform()
            title_transform.translate(self.width() / 2 - .5 * self._title_width,
                                      self.height() - self._title_height - self.margin_bottom)

            self.margin_bottom = self._maxLabelWidth / 2 + self._labelMargin
            self.gradient.setFinalStop(self.width() - 2 * self.margin_bottom, 0)
            self._barRect.setHeight(self._barThickness)
            self._barRect.setLeft(self.margin_bottom)  # reduces width by margin (as opposed to shifting)
            self._barRect.setWidth(self._barRect.width() - self.margin_bottom)

            painter.fillRect(self._barRect, self.gradient)

            line_step = self._barRect.width() / 20
            pos = self._barRect.left()
            while pos <= self._barRect.right():
                painter.drawLine(pos, 2 * (self._barRect.height()) / 3,
                                 pos, self._barRect.bottom())
                pos += line_step

        if self._title is not None:
            title_doc = QTextDocument()
            title_doc.setDefaultFont(self.font())
            title_doc.setDocumentMargin(0)
            title_doc.setHtml(self._title)

            painter.save()
            painter.setTransform(title_transform)
            title_doc.drawContents(painter)
            painter.restore()

        label_doc = QTextDocument()
        label_doc.setDefaultFont(self.font())
        label_doc.setDocumentMargin(0)
        for pos, label in self.labels.items():
            label_doc.setHtml(label)
            # Figure coordinate position. 1=height-margin for vertical, or width-margin for horizontal

            if self._orientation == 'Vertical':
                lpos = (-1 * (self._barRect.height() * pos))
                l_offset = (.5 * self._maxLabelHeight)
                pos = QPointF(self._barThickness + self._labelMargin,
                              lpos - l_offset)
            elif self._orientation == 'Horizontal':
                text_width = label_doc.idealWidth()
                lpos = ((self.width() - (2 * self.margin_bottom)) * pos)  # Center position
                lleft = lpos - text_width / 2
                pos = QPointF(lleft + self.margin_bottom, self._barThickness + 3)

            painter.save()
            painter.translate(pos)
            label_doc.drawContents(painter)
            painter.restore()

    def setTitle(self, title):
        self._title = title
        self._calc_label_sizes()
        self.repaint()

    def setGradient(self, g):
        self.gradient = g
        self.repaint()

    def setLabels(self, l):
        self.labels = l
        self._calc_label_sizes()
        self.repaint()

    def _calc_label_sizes(self):
        if not self.labels:
            self._maxLabelHeight = 0
            self._maxLabelWidth = 0
            return

        metrics = QFontMetrics(self.font())
        if self._orientation == 'Vertical':
            to_check = self.labels.values()
        else:
            # If we are horizontal, we only need the max width of the first and last label.
            # Technically, we only even need that if they are at the ends.
            min_label = min(self.labels.keys())
            max_label = max(self.labels.keys())
            to_check = (self.labels[min_label], self.labels[max_label])

        self._maxLabelWidth = 0
        self._maxLabelHeight = 0
        for k in to_check:
            b = metrics.boundingRect(str(k))
            self._maxLabelWidth = max(self._maxLabelWidth, b.width())
            self._maxLabelHeight = max(self._maxLabelHeight, b.height())

        self._title_height = 0
        if self._title is not None:
            title_doc = QTextDocument()
            title_doc.setDefaultFont(self.font())
            title_doc.setDocumentMargin(0)
            title_doc.setHtml(self._title)
            self._title_width = title_doc.idealWidth()
            title_doc.setTextWidth(self._title_width)
            self._title_height = title_doc.size().height()

        # Font metrics are overly generous (at least for width),
        # so decrease the calculated values significantly
        self._maxLabelWidth *= .8

        if self._orientation == 'Horizontal':
            self.setMinimumHeight(self._maxLabelHeight + self._barThickness + self._labelMargin + self._title_height)
        elif self._orientation == "Vertical":
            self.setMinimumWidth(self._barThickness + self._labelMargin + self._maxLabelWidth + self._title_height)

    def setOrientation(self, orientation):
        if not orientation in ('Horizontal', 'Vertical'):
            raise TypeError("Orientation must be one of 'Horizontal' or 'Vertical'")

        self._orientation = orientation
        self._calc_label_sizes()
        self.repaint()

    def setFont(self, font):
        super().setFont(font)
        self._calc_label_sizes()
