from __future__ import annotations

from pathlib import Path

import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QTableWidgetItem,
)

from frontend.components.file_table import FileTable

from backend.api.app_state import AppState
from backend.api.routes import preview_output


class AnalyticsView(QWidget):

    def __init__(self, state: AppState):
        super().__init__()

        self.state = state

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<h2>Analytics</h2>"))

        self.summary = QLabel(
            "Run the pipeline to view analytics."
        )

        layout.addWidget(self.summary)

        self.table = FileTable()
        layout.addWidget(self.table)

        self.refresh_button = QPushButton("Refresh Preview")
        self.refresh_button.clicked.connect(self.refresh)

        layout.addWidget(self.refresh_button)

        self.output_button = QPushButton("Open Output Excel")
        self.output_button.clicked.connect(self.open_output)

        layout.addWidget(self.output_button)

    def refresh(self):

        if self.state.current_output_file is None:
            self.summary.setText("No output has been generated.")
            return

        summary = self.state.last_summary

        if summary is not None:
            self.summary.setText(
                f"""
Trials Processed: {summary.trials_processed}

Criteria Extracted: {summary.criteria_extracted}

After Normalization: {summary.criteria_after_normalization}

After Deduplication: {summary.criteria_after_deduplication}

Estimated Cost: ${summary.total_cost_usd:.4f}
                """
            )

        df = preview_output(
            self.state.current_output_file,
            n=25,
        )

        self.populate_table(df)

    def populate_table(self, df):
        self.table.load_dataframe(df)

    def open_output(self):

        if self.state.current_output_file is None:
            return

        path = Path(self.state.current_output_file)

        if path.exists():
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(path))
            )