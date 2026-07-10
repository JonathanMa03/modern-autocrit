from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from backend.api.app_state import AppState
from backend.api.routes import run_pipeline


class PipelineWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        state: AppState,
        xml_directory: Path,
        output_file: Path,
    ):
        super().__init__()

        self.state = state
        self.xml_directory = xml_directory
        self.output_file = output_file

    @Slot()
    def run(self) -> None:
        try:
            summary = run_pipeline(
                state=self.state,
                xml_directory=self.xml_directory,
                output_excel=self.output_file,
            )
            self.finished.emit(summary)

        except Exception as exc:
            self.failed.emit(str(exc))