from __future__ import annotations

import logging
import queue
import re
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from backend.api.app_state import AppState
from backend.api.routes import run_pipeline


class QueueLogHandler(logging.Handler):
    """
    Send Python log records into a thread-safe queue.
    """

    def __init__(self, log_queue: queue.Queue[str]) -> None:
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.log_queue.put(self.format(record))
        except Exception:
            self.handleError(record)


class RunView(ttk.Frame):
    """
    Tkinter Run tab for launching the Modern AutoCrit pipeline.
    """

    TRIAL_FINISHED_PATTERN = re.compile(
        r"\bNCT\d+\s+finished with\s+\d+\s+extracted criteria",
        flags=re.IGNORECASE,
    )

    def __init__(
        self,
        parent: tk.Widget,
        state: AppState,
    ) -> None:
        super().__init__(parent, padding=15)

        self.state = state

        self.xml_directory = tk.StringVar()
        self.output_file = tk.StringVar()
        self.status_text = tk.StringVar(value="Ready")
        self.progress_text = tk.StringVar(value="0%")
        self.progress_value = tk.DoubleVar(value=0.0)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.result_queue: queue.Queue[dict] = queue.Queue()

        self.pipeline_thread: threading.Thread | None = None

        self.total_trials = 0
        self.completed_trials = 0
        self.logs_visible = False

        self._build_ui()
        self._configure_logging()

        self.after(100, self._poll_queues)

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(7, weight=1)

        title = ttk.Label(
            self,
            text="Run Modern AutoCrit",
            style="Section.TLabel",
        )
        title.grid(
            row=0,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(0, 15),
        )

        # XML directory
        ttk.Label(
            self,
            text="XML Folder",
        ).grid(
            row=1,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=5,
        )

        self.xml_entry = ttk.Entry(
            self,
            textvariable=self.xml_directory,
        )
        self.xml_entry.grid(
            row=1,
            column=1,
            sticky="ew",
            pady=5,
        )

        ttk.Button(
            self,
            text="Browse...",
            command=self.select_xml_folder,
        ).grid(
            row=1,
            column=2,
            padx=(10, 0),
            pady=5,
        )

        # Output path
        ttk.Label(
            self,
            text="Output Excel",
        ).grid(
            row=2,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=5,
        )

        self.output_entry = ttk.Entry(
            self,
            textvariable=self.output_file,
        )
        self.output_entry.grid(
            row=2,
            column=1,
            sticky="ew",
            pady=5,
        )

        ttk.Button(
            self,
            text="Browse...",
            command=self.select_output_file,
        ).grid(
            row=2,
            column=2,
            padx=(10, 0),
            pady=5,
        )

        # Controls
        controls = ttk.Frame(self)
        controls.grid(
            row=3,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(15, 10),
        )

        self.run_button = ttk.Button(
            controls,
            text="Run Pipeline",
            command=self.start_pipeline,
        )
        self.run_button.pack(side="left")

        self.log_toggle_button = ttk.Button(
            controls,
            text="Show Logs",
            command=self.toggle_logs,
        )
        self.log_toggle_button.pack(side="left", padx=(10, 0))

        self.clear_button = ttk.Button(
            controls,
            text="Clear Logs",
            command=self.clear_log,
        )
        self.clear_button.pack(side="left", padx=(10, 0))

        ttk.Label(
            controls,
            textvariable=self.status_text,
        ).pack(side="right")

        # Progress bar and percentage
        progress_frame = ttk.Frame(self)
        progress_frame.grid(
            row=4,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(0, 10),
        )
        progress_frame.columnconfigure(0, weight=1)

        self.progress = ttk.Progressbar(
            progress_frame,
            mode="determinate",
            maximum=100,
            variable=self.progress_value,
        )
        self.progress.grid(
            row=0,
            column=0,
            sticky="ew",
        )

        ttk.Label(
            progress_frame,
            textvariable=self.progress_text,
            width=6,
            anchor="e",
        ).grid(
            row=0,
            column=1,
            padx=(10, 0),
        )

        self.progress_detail = ttk.Label(
            self,
            text="No pipeline is running.",
        )
        self.progress_detail.grid(
            row=5,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(0, 10),
        )

        # Log section
        self.log_frame = ttk.Frame(self)
        self.log_frame.grid(
            row=6,
            column=0,
            columnspan=3,
            rowspan=2,
            sticky="nsew",
        )
        self.log_frame.columnconfigure(0, weight=1)
        self.log_frame.rowconfigure(1, weight=1)

        ttk.Label(
            self.log_frame,
            text="Live Logs",
            style="Section.TLabel",
        ).grid(
            row=0,
            column=0,
            sticky="w",
            pady=(5, 5),
        )

        self.console = ScrolledText(
            self.log_frame,
            wrap="none",
            height=24,
            state="disabled",
            font=("Menlo", 11),
        )
        self.console.grid(
            row=1,
            column=0,
            sticky="nsew",
        )

        # Hidden by default.
        self.log_frame.grid_remove()

    def _configure_logging(self) -> None:
        self.log_handler = QueueLogHandler(self.log_queue)
        self.log_handler.setLevel(logging.INFO)
        self.log_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )

        root_logger = logging.getLogger()

        for handler in list(root_logger.handlers):
            if isinstance(handler, QueueLogHandler):
                root_logger.removeHandler(handler)

        root_logger.addHandler(self.log_handler)
        root_logger.setLevel(logging.INFO)

    def select_xml_folder(self) -> None:
        folder = filedialog.askdirectory(
            title="Select ClinicalTrials.gov XML Folder",
        )

        if folder:
            self.xml_directory.set(folder)

            count = self._count_xml_files(Path(folder))
            self.progress_detail.configure(
                text=f"{count} XML trial file(s) detected."
            )

    def select_output_file(self) -> None:
        filename = filedialog.asksaveasfilename(
            title="Select Output Excel File",
            defaultextension=".xlsx",
            filetypes=[
                ("Excel files", "*.xlsx"),
            ],
            initialfile="modern_autocrit_output.xlsx",
        )

        if filename:
            self.output_file.set(filename)

    @staticmethod
    def _count_xml_files(xml_directory: Path) -> int:
        """
        Count XML files recursively so nested trial folders are supported.
        """
        if not xml_directory.exists():
            return 0

        return sum(
            1
            for path in xml_directory.rglob("*.xml")
            if path.is_file()
        )

    def start_pipeline(self) -> None:
        xml_directory_text = self.xml_directory.get().strip()
        output_file_text = self.output_file.get().strip()

        if not xml_directory_text:
            messagebox.showwarning(
                "Missing XML Folder",
                "Select an XML input folder.",
            )
            return

        if not output_file_text:
            messagebox.showwarning(
                "Missing Output File",
                "Select an output Excel file.",
            )
            return

        xml_directory = Path(xml_directory_text)
        output_file = Path(output_file_text)

        if not xml_directory.exists():
            messagebox.showerror(
                "Invalid XML Folder",
                "The selected XML folder does not exist.",
            )
            return

        self.total_trials = self._count_xml_files(xml_directory)

        if self.total_trials == 0:
            messagebox.showwarning(
                "No XML Files",
                "No XML files were found in the selected folder.",
            )
            return

        if self.pipeline_thread and self.pipeline_thread.is_alive():
            return

        self.completed_trials = 0
        self.progress_value.set(0)
        self.progress_text.set("0%")
        self.progress_detail.configure(
            text=f"Processing 0 of {self.total_trials} trials."
        )

        self.clear_log()
        self.append_log("Starting Modern AutoCrit pipeline...")

        self.status_text.set("Running")
        self.run_button.configure(state="disabled")

        self.pipeline_thread = threading.Thread(
            target=self._run_pipeline_worker,
            args=(xml_directory, output_file),
            daemon=True,
        )
        self.pipeline_thread.start()

    def _run_pipeline_worker(
        self,
        xml_directory: Path,
        output_file: Path,
    ) -> None:
        try:
            summary = run_pipeline(
                state=self.state,
                xml_directory=xml_directory,
                output_excel=output_file,
            )

            self.result_queue.put(
                {
                    "status": "success",
                    "summary": summary,
                }
            )

        except Exception as exc:
            logging.getLogger(__name__).exception(
                "Pipeline failed."
            )

            self.result_queue.put(
                {
                    "status": "error",
                    "error": str(exc),
                }
            )

    def _poll_queues(self) -> None:
        self._poll_log_queue()
        self._poll_result_queue()

        self.after(100, self._poll_queues)

    def _poll_log_queue(self) -> None:
        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break

            self.append_log(message)
            self._update_progress_from_log(message)

    def _update_progress_from_log(self, message: str) -> None:
        """
        Update percentage when the pipeline logs that a trial finished.

        Example expected log:
        NCT00114192 finished with 35 extracted criteria.
        """
        if not self.TRIAL_FINISHED_PATTERN.search(message):
            return

        self.completed_trials = min(
            self.completed_trials + 1,
            self.total_trials,
        )

        percentage = (
            self.completed_trials / self.total_trials
        ) * 100

        self.progress_value.set(percentage)
        self.progress_text.set(f"{percentage:.0f}%")
        self.progress_detail.configure(
            text=(
                f"Processing {self.completed_trials} "
                f"of {self.total_trials} trials."
            )
        )

    def _poll_result_queue(self) -> None:
        while True:
            try:
                result = self.result_queue.get_nowait()
            except queue.Empty:
                break

            if result["status"] == "success":
                self._handle_success(result["summary"])
            else:
                self._handle_error(result["error"])

    def _handle_success(self, summary) -> None:
        self.progress_value.set(100)
        self.progress_text.set("100%")
        self.progress_detail.configure(
            text=(
                f"Completed {summary.trials_processed} "
                "trial(s)."
            )
        )

        self.run_button.configure(state="normal")
        self.status_text.set("Completed")

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
            f"Output file: {summary.output_file}"
        )

        messagebox.showinfo(
            "Pipeline Complete",
            "Modern AutoCrit finished successfully.",
        )

    def _handle_error(self, error_message: str) -> None:
        self.run_button.configure(state="normal")
        self.status_text.set("Failed")

        self.progress_detail.configure(
            text=(
                f"Stopped after {self.completed_trials} "
                f"of {self.total_trials} trials."
            )
        )

        self.append_log("")
        self.append_log(f"Pipeline failed: {error_message}")

        messagebox.showerror(
            "Pipeline Failed",
            error_message,
        )

    def toggle_logs(self) -> None:
        if self.logs_visible:
            self.log_frame.grid_remove()
            self.log_toggle_button.configure(text="Show Logs")
            self.logs_visible = False
        else:
            self.log_frame.grid()
            self.log_toggle_button.configure(text="Hide Logs")
            self.logs_visible = True

    def append_log(self, message: str) -> None:
        self.console.configure(state="normal")
        self.console.insert("end", message + "\n")
        self.console.see("end")
        self.console.configure(state="disabled")

    def clear_log(self) -> None:
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    def destroy(self) -> None:
        logging.getLogger().removeHandler(self.log_handler)
        super().destroy()