from __future__ import annotations

import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QTableWidget,
    QTableWidgetItem,
)


class FileTable(QTableWidget):
    """
    Generic read-only DataFrame viewer.

    Used throughout the application for:
        • pipeline output preview
        • unmapped terminology
        • semantic duplicate candidates
        • dictionary editor
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)

    def load_dataframe(
        self,
        df: pd.DataFrame,
    ) -> None:

        self.clear()

        if df is None or df.empty:
            self.setRowCount(0)
            self.setColumnCount(0)
            return

        self.setRowCount(len(df))
        self.setColumnCount(len(df.columns))
        self.setHorizontalHeaderLabels(df.columns.astype(str))

        for r in range(len(df)):
            for c in range(len(df.columns)):
                value = df.iat[r, c]

                if pd.isna(value):
                    value = ""

                item = QTableWidgetItem(str(value))

                item.setFlags(
                    item.flags() & ~Qt.ItemIsEditable
                )

                self.setItem(r, c, item)

        self.resizeColumnsToContents()
        self.resizeRowsToContents()

    def clear_table(self):
        self.clear()
        self.setRowCount(0)
        self.setColumnCount(0)