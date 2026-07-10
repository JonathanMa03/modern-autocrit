from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from backend.api.app_state import AppState
from frontend.workers.pipeline_worker import PipelineWorker


class LogEmitter(QObject):
    message = Signal(str)


class QtLogHandler(logging.Handler):
    """
    Sends Python logging records to a Qt signal.
    """

    def __init__(self, emitter: LogEmitter):
        super().__init__()
        self.emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            self.emitter.message.emit(message)
        except Exception:
            self.handleError(record)


class RunView(QWidget):
    def __init__(self, state: AppState):
        super().__init__()

        self.state = state
        self.xml_directory: Path | None = None
        self.output_file: Path | None = None

        self.thread: QThread | None = None
        self.worker: PipelineWorker | None = None

        self._build_ui()
        self._configure_logging()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<h2>Run Modern AutoCrit</h2>"))

        # XML input directory
        xml_layout = QHBoxLayout()

        xml_button = QPushButton("Select XML Folder")
        xml_button.clicked.connect(self.select_xml_folder)

        self.xml_label = QLabel("No XML folder selected")

        xml_layout.addWidget(xml_button)
        xml_layout.addWidget(self.xml_label, stretch=1)
        layout.addLayout(xml_layout)

        # Output file
        output_layout = QHBoxLayout()

        output_button = QPushButton("Select Output File")
        output_button.clicked.connect(self.select_output_file)

        self.output_label = QLabel("No output file selected")

        output_layout.addWidget(output_button)
        output_layout.addWidget(self.output_label, stretch=1)
        layout.addLayout(output_layout)

        # Run controls
        controls = QHBoxLayout()

        self.run_button = QPushButton("Run Extraction")
        self.run_button.clicked.connect(self.start_pipeline)

        self.clear_button = QPushButton("Clear Log")
        self.clear_button.clicked.connect(self.clear_log)

        controls.addWidget(self.run_button)
        controls.addWidget(self.clear_button)
        controls.addStretch()

        layout.addLayout(controls)

        # Indeterminate progress bar while running
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        # Live log console
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setLineWrapMode(QTextEdit.NoWrap)
        layout.addWidget(self.console)

    def _configure_logging(self) -> None:
        self.log_emitter = LogEmitter()
        self.log_emitter.message.connect(self.append_log)

        self.log_handler = QtLogHandler(self.log_emitter)
        self.log_handler.setLevel(logging.INFO)
        self.log_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )

        root_logger = logging.getLogger()

        # Prevent duplicate GUI handlers if the view is reconstructed.
        for handler in list(root_logger.handlers):
            if isinstance(handler, QtLogHandler):
                root_logger.removeHandler(handler)

        root_logger.addHandler(self.log_handler)
        root_logger.setLevel(logging.INFO)

    def select_xml_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select ClinicalTrials.gov XML Folder",
        )

        if folder:
            self.xml_directory = Path(folder)
            self.xml_label.setText(folder)

    def select_output_file(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Select Output Excel File",
            "modern_autocrit_output.xlsx",
            "Excel files (*.xlsx)",
        )

        if filename:
            path = Path(filename)

            if path.suffix.lower() != ".xlsx":
                path = path.with_suffix(".xlsx")

            self.output_file = path
            self.output_label.setText(str(path))

    def start_pipeline(self) -> None:
        if self.xml_directory is None:
            QMessageBox.warning(
                self,
                "Missing XML Folder",
                "Select an XML input folder before running extraction.",
            )
            return

        if self.output_file is None:
            QMessageBox.warning(
                self,
                "Missing Output File",
                "Select an output Excel file before running extraction.",
            )
            return

        if self.thread is not None and self.thread.isRunning():
            return

        self.console.clear()
        self.append_log("Starting Modern AutoCrit pipeline…")

        self.run_button.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Indeterminate animation
        self.status_label.setText("Running extraction…")

        self.thread = QThread(self)

        self.worker = PipelineWorker(
            state=self.state,
            xml_directory=self.xml_directory,
            output_file=self.output_file,
        )

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)

        self.worker.finished.connect(self.pipeline_finished)
        self.worker.failed.connect(self.pipeline_failed)

        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)

        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.worker.deleteLater)

        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._clear_worker_references)

        self.thread.start()

    def pipeline_finished(self, summary: object) -> None:
        self.append_log("")
        self.append_log("Pipeline finished successfully.")
        self.append_log(
            f"Trials processed: {summary.trials_processed}"
        )
        self.append_log(
            f"Criteria extracted: {summary.criteria_extracted}"
        )
        self.append_log(
            "After normalization: "
            f"{summary.criteria_after_normalization}"
        )
        self.append_log(
            "After deduplication: "
            f"{summary.criteria_after_deduplication}"
        )
        self.append_log(
            f"Estimated cost: ${summary.total_cost_usd:.6f}"
        )
        self.append_log(
            f"Output: {summary.output_file}"
        )

        self.status_label.setText("Completed")
        self._set_idle_state()

    def pipeline_failed(self, error_message: str) -> None:
        self.append_log("")
        self.append_log(f"Pipeline failed: {error_message}")

        self.status_label.setText("Failed")
        self._set_idle_state()

        QMessageBox.critical(
            self,
            "Pipeline Failed",
            error_message,
        )

    def _set_idle_state(self) -> None:
        self.run_button.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setVisible(False)

    def _clear_worker_references(self) -> None:
        self.thread = None
        self.worker = None

    def append_log(self, message: str) -> None:
        self.console.append(message)

        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_log(self) -> None:
        self.console.clear()

    def closeEvent(self, event) -> None:
        logging.getLogger().removeHandler(self.log_handler)

        if self.thread is not None and self.thread.isRunning():
            event.ignore()
            QMessageBox.warning(
                self,
                "Pipeline Running",
                "Wait for the current extraction to finish before closing.",
            )
            return

        event.accept()