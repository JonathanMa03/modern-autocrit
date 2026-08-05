from __future__ import annotations

import queue
import re
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from backend.api.app_state import AppState
from backend.validation_agent import IntegrityValidationAgent, ValidationRunResult


class IntegrityView(ttk.Frame):
    """Interactive interface for independent extraction-integrity audits."""

    def __init__(self, parent: tk.Widget, state: AppState) -> None:
        super().__init__(parent, padding=15)
        self.state = state
        self.xml_directory_var = tk.StringVar(value="data/raw/xml_trials")
        self.extraction_file_var = tk.StringVar(
            value=str(state.current_output_file or "outputs/modern_autocrit_output.xlsx")
        )
        self.report_file_var = tk.StringVar(
            value="outputs/integrity_validation.xlsx"
        )
        self.trial_ids_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Select inputs and run an integrity audit.")
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.agent: IntegrityValidationAgent | None = None
        self.last_run: ValidationRunResult | None = None
        self._build_ui()
        self.after(100, self._poll_events)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

        ttk.Label(self, text="Integrity", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 12)
        )

        inputs = ttk.LabelFrame(self, text="Validation Inputs", padding=10)
        inputs.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        inputs.columnconfigure(1, weight=1)
        self._path_row(
            inputs,
            0,
            "Trial XML Folder",
            self.xml_directory_var,
            self._select_xml_directory,
        )
        self._path_row(
            inputs,
            1,
            "Extraction Workbook",
            self.extraction_file_var,
            self._select_extraction_file,
        )
        self._path_row(
            inputs,
            2,
            "Validation Report",
            self.report_file_var,
            self._select_report_file,
        )
        ttk.Label(inputs, text="Trial IDs").grid(
            row=3, column=0, sticky="w", padx=(0, 10), pady=4
        )
        ttk.Entry(inputs, textvariable=self.trial_ids_var).grid(
            row=3, column=1, sticky="ew", pady=4
        )
        ttk.Label(
            inputs,
            text="Optional; comma-separated NCT IDs. Leave blank to audit all XML files.",
        ).grid(row=3, column=2, sticky="w", padx=(10, 0))

        controls = ttk.Frame(self)
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.run_button = ttk.Button(
            controls, text="Run Validation", command=self._start_validation
        )
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(
            controls,
            text="Stop",
            command=self._stop_validation,
            state="disabled",
        )
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Label(controls, textvariable=self.status_var).pack(side="left", padx=12)

        progress_frame = ttk.Frame(self)
        progress_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(
            progress_frame,
            mode="determinate",
            maximum=100,
        )
        self.progress.grid(row=0, column=0, sticky="ew")
        self.progress_label = ttk.Label(progress_frame, text="0 / 0", width=18)
        self.progress_label.grid(row=0, column=1, padx=(10, 0))

        chat_frame = ttk.LabelFrame(self, text="Validation Agent", padding=10)
        chat_frame.grid(row=4, column=0, sticky="nsew")
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        self.chat = ScrolledText(chat_frame, wrap="word", state="disabled")
        self.chat.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.question = tk.Text(chat_frame, height=3, wrap="word")
        self.question.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.question.bind("<Control-Return>", self._send_from_event)
        self.send_button = ttk.Button(
            chat_frame,
            text="Send",
            command=self._start_chat,
            state="disabled",
        )
        self.send_button.grid(row=1, column=1, sticky="ns", padx=(8, 0), pady=(8, 0))
        self._append_chat(
            "Integrity Agent",
            "Run validation to compare the source eligibility text with the extraction workbook. "
            "Afterward, ask questions about missed, unsupported, or partial criteria.",
        )

    @staticmethod
    def _path_row(parent, row, label, variable, command) -> None:
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 10), pady=4
        )
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", pady=4
        )
        ttk.Button(parent, text="Browse...", command=command).grid(
            row=row, column=2, padx=(10, 0), pady=4
        )

    def _select_xml_directory(self) -> None:
        path = filedialog.askdirectory(title="Select trial XML folder")
        if path:
            self.xml_directory_var.set(path)

    def _select_extraction_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select extraction workbook",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if path:
            self.extraction_file_var.set(path)

    def _select_report_file(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save validation report",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile="integrity_validation.xlsx",
        )
        if path:
            self.report_file_var.set(path)

    def _start_validation(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        xml_directory = Path(self.xml_directory_var.get().strip())
        extraction_file = Path(self.extraction_file_var.get().strip())
        report_file = Path(self.report_file_var.get().strip())
        if not xml_directory.is_dir():
            messagebox.showerror("Integrity", "Select a valid trial XML folder.")
            return
        if not extraction_file.is_file():
            messagebox.showerror("Integrity", "Select a valid extraction workbook.")
            return
        trial_ids = [
            item.upper()
            for item in re.split(r"[,;\s]+", self.trial_ids_var.get().strip())
            if item
        ]
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.send_button.configure(state="disabled")
        self.stop_event = threading.Event()
        self.progress.configure(value=0)
        self.progress_label.configure(text="0 / 0")
        self.status_var.set("Validation is running…")
        self._append_chat(
            "Integrity Agent",
            f"Starting an independent audit for {'the selected trials' if trial_ids else 'all XML trials'}.",
        )

        def work() -> None:
            try:
                agent = IntegrityValidationAgent(
                    self.state.settings.llm,
                    self.state.cost_monitor,
                )
                result = agent.validate(
                    xml_directory,
                    extraction_file,
                    trial_ids=trial_ids,
                    report_file=report_file,
                    stop_event=self.stop_event,
                    progress_callback=lambda completed, total, trial_id: self.events.put(
                        ("progress", (completed, total, trial_id))
                    ),
                )
                self.events.put(("validation_complete", (agent, result)))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _stop_validation(self) -> None:
        if not self.worker or not self.worker.is_alive():
            return
        self.stop_event.set()
        self.stop_button.configure(state="disabled")
        self.status_var.set("Stopping after the current trial…")
        self._append_chat(
            "Integrity Agent",
            "Stop requested. The current model request will finish, but no additional trial will start.",
        )

    def _start_chat(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        question = self.question.get("1.0", "end").strip()
        if not question or not self.agent or not self.last_run:
            return
        self.question.delete("1.0", "end")
        self._append_chat("You", question)
        self.send_button.configure(state="disabled")
        self.status_var.set("Integrity agent is responding…")

        def work() -> None:
            try:
                answer = self.agent.chat(self.last_run, question)
                self.events.put(("chat_complete", answer))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _send_from_event(self, _event):
        self._start_chat()
        return "break"

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "validation_complete":
                    self.agent, self.last_run = payload
                    self._show_validation_summary(self.last_run)
                    self.run_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.send_button.configure(
                        state="normal" if self.last_run.trials else "disabled"
                    )
                elif event == "chat_complete":
                    self._append_chat("Integrity Agent", str(payload))
                    self.send_button.configure(state="normal")
                    self.status_var.set("Ready for follow-up questions.")
                elif event == "progress":
                    completed, total, trial_id = payload
                    percent = (completed / total * 100) if total else 0
                    self.progress.configure(value=percent)
                    label = f"{completed} / {total}"
                    if trial_id and completed < total:
                        label += f" · {trial_id}"
                    self.progress_label.configure(text=label)
                elif event == "error":
                    self._append_chat("Error", str(payload))
                    self.status_var.set("Validation failed.")
                    self.run_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.send_button.configure(
                        state="normal" if self.last_run else "disabled"
                    )
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _show_validation_summary(self, run: ValidationRunResult) -> None:
        if not run.trials:
            message = "No trials were successfully validated."
        else:
            total_clauses = sum(t.metrics.total_source_clauses for t in run.trials)
            missed = sum(t.metrics.uncovered_clauses for t in run.trials)
            unsupported = sum(t.metrics.unsupported_rows for t in run.trials)
            average_f1 = sum(t.metrics.estimated_f1 for t in run.trials) / len(run.trials)
            message = (
                f"Validated {len(run.trials)} trial(s): {total_clauses} independently identified "
                f"criteria, {missed} uncovered criteria, {unsupported} unsupported extraction(s), "
                f"estimated mean F1 {average_f1:.1%}."
            )
        if run.errors:
            message += "\n\nWarnings:\n- " + "\n- ".join(run.errors)
        if run.report_file:
            message += f"\n\nReport saved to {run.report_file}."
        if run.cancelled:
            message += "\n\nValidation was stopped by the user; the report contains partial results."
        self._append_chat("Integrity Agent", message)
        self.status_var.set(
            "Validation stopped."
            if run.cancelled
            else ("Validation complete." if run.trials else "Validation completed with no results.")
        )

    def _append_chat(self, speaker: str, message: str) -> None:
        self.chat.configure(state="normal")
        self.chat.insert("end", f"{speaker}:\n{message}\n\n")
        self.chat.configure(state="disabled")
        self.chat.see("end")
