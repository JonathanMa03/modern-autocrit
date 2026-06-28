from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QTextEdit,
)

from backend.api.routes import run_pipeline
from backend.api.app_state import AppState


class RunView(QWidget):
    def __init__(self, state: AppState):
        super().__init__()

        self.state = state

        self.xml_directory: Path | None = None
        self.output_file: Path | None = None

        layout = QVBoxLayout(self)

        title = QLabel("<h2>Run Modern AutoCrit</h2>")
        layout.addWidget(title)

        # -----------------------
        # XML folder
        # -----------------------

        xml_layout = QHBoxLayout()

        self.xml_label = QLabel("No XML folder selected")

        xml_button = QPushButton("Select XML Folder")
        xml_button.clicked.connect(self.select_xml_folder)

        xml_layout.addWidget(xml_button)
        xml_layout.addWidget(self.xml_label)

        layout.addLayout(xml_layout)

        # -----------------------
        # Output file
        # -----------------------

        out_layout = QHBoxLayout()

        self.output_label = QLabel("No output file selected")

        out_button = QPushButton("Select Output File")
        out_button.clicked.connect(self.select_output_file)

        out_layout.addWidget(out_button)
        out_layout.addWidget(self.output_label)

        layout.addLayout(out_layout)

        # -----------------------
        # Run button
        # -----------------------

        self.run_button = QPushButton("Run Extraction")
        self.run_button.clicked.connect(self.run_pipeline)

        layout.addWidget(self.run_button)

        # -----------------------
        # Console
        # -----------------------

        self.console = QTextEdit()
        self.console.setReadOnly(True)

        layout.addWidget(self.console)

    def log(self, text: str):
        self.console.append(text)

    def select_xml_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select XML Folder",
        )

        if folder:
            self.xml_directory = Path(folder)
            self.xml_label.setText(folder)

    def select_output_file(self):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Output Excel",
            "modern_autocrit_output.xlsx",
            "Excel (*.xlsx)",
        )

        if filename:
            self.output_file = Path(filename)
            self.output_label.setText(filename)

    def run_pipeline(self):
        if self.xml_directory is None:
            self.log("Please select an XML folder.")
            return

        if self.output_file is None:
            self.log("Please select an output file.")
            return

        self.log("Running pipeline...")

        try:
            summary = run_pipeline(
                self.state,
                self.xml_directory,
                self.output_file,
            )

            self.log("")
            self.log("Finished.")
            self.log(f"Trials processed: {summary.trials_processed}")
            self.log(f"Criteria extracted: {summary.criteria_extracted}")
            self.log(f"After normalization: {summary.criteria_after_normalization}")
            self.log(f"After deduplication: {summary.criteria_after_deduplication}")
            self.log(f"Cost: ${summary.total_cost_usd:.4f}")

        except Exception as exc:
            self.log("")
            self.log(str(exc))