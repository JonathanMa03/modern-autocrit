from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from backend.api.app_state import AppState
from backend.api.routes import preview_output


class AnalyticsView(ttk.Frame):
    """
    Analytics tab for reviewing the latest pipeline run.

    Features:
        - pipeline summary
        - output preview
        - unmapped terminology preview
        - refresh controls
        - open output files
        - load an existing output workbook manually
    """

    def __init__(
        self,
        parent: tk.Widget,
        state: AppState,
    ) -> None:
        super().__init__(parent, padding=15)

        self.state = state

        self.output_file_var = tk.StringVar()
        self.status_var = tk.StringVar(
            value="Run the pipeline or load an output file."
        )

        self.trials_var = tk.StringVar(value="—")
        self.extracted_var = tk.StringVar(value="—")
        self.normalized_var = tk.StringVar(value="—")
        self.deduplicated_var = tk.StringVar(value="—")
        self.cost_var = tk.StringVar(value="—")

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

        ttk.Label(
            self,
            text="Analytics",
            style="Section.TLabel",
        ).grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 15),
        )

        # -------------------------------------------------
        # File controls
        # -------------------------------------------------

        file_frame = ttk.LabelFrame(
            self,
            text="Output File",
            padding=12,
        )
        file_frame.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(0, 12),
        )
        file_frame.columnconfigure(1, weight=1)

        ttk.Label(
            file_frame,
            text="Workbook",
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 10),
        )

        ttk.Entry(
            file_frame,
            textvariable=self.output_file_var,
        ).grid(
            row=0,
            column=1,
            sticky="ew",
        )

        ttk.Button(
            file_frame,
            text="Browse...",
            command=self.select_output_file,
        ).grid(
            row=0,
            column=2,
            padx=(10, 0),
        )

        ttk.Button(
            file_frame,
            text="Refresh",
            command=self.refresh,
        ).grid(
            row=0,
            column=3,
            padx=(10, 0),
        )

        ttk.Button(
            file_frame,
            text="Open Output",
            command=self.open_output_file,
        ).grid(
            row=0,
            column=4,
            padx=(10, 0),
        )

        ttk.Button(
            file_frame,
            text="Open Unmapped Terms",
            command=self.open_unmapped_file,
        ).grid(
            row=0,
            column=5,
            padx=(10, 0),
        )

        # -------------------------------------------------
        # Summary cards
        # -------------------------------------------------

        summary_frame = ttk.LabelFrame(
            self,
            text="Pipeline Summary",
            padding=12,
        )
        summary_frame.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(0, 12),
        )

        for column in range(5):
            summary_frame.columnconfigure(
                column,
                weight=1,
            )

        self._create_summary_card(
            summary_frame,
            column=0,
            label="Trials",
            variable=self.trials_var,
        )
        self._create_summary_card(
            summary_frame,
            column=1,
            label="Extracted",
            variable=self.extracted_var,
        )
        self._create_summary_card(
            summary_frame,
            column=2,
            label="Normalized",
            variable=self.normalized_var,
        )
        self._create_summary_card(
            summary_frame,
            column=3,
            label="Deduplicated",
            variable=self.deduplicated_var,
        )
        self._create_summary_card(
            summary_frame,
            column=4,
            label="Estimated Cost",
            variable=self.cost_var,
        )

        # -------------------------------------------------
        # Preview tabs
        # -------------------------------------------------

        preview_notebook = ttk.Notebook(self)
        preview_notebook.grid(
            row=4,
            column=0,
            sticky="nsew",
        )

        self.output_preview_tab = ttk.Frame(
            preview_notebook,
        )
        self.unmapped_preview_tab = ttk.Frame(
            preview_notebook,
        )

        preview_notebook.add(
            self.output_preview_tab,
            text="Output Preview",
        )
        preview_notebook.add(
            self.unmapped_preview_tab,
            text="Unmapped Terms",
        )

        self.output_tree = self._create_table(
            self.output_preview_tab,
        )
        self.unmapped_tree = self._create_table(
            self.unmapped_preview_tab,
        )

        ttk.Label(
            self,
            textvariable=self.status_var,
        ).grid(
            row=5,
            column=0,
            sticky="w",
            pady=(10, 0),
        )

    def _create_summary_card(
        self,
        parent: ttk.Frame,
        column: int,
        label: str,
        variable: tk.StringVar,
    ) -> None:
        frame = ttk.Frame(
            parent,
            padding=8,
        )
        frame.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=5,
        )

        ttk.Label(
            frame,
            text=label,
        ).pack()

        ttk.Label(
            frame,
            textvariable=variable,
            font=("Helvetica", 16, "bold"),
        ).pack(pady=(4, 0))

    def _create_table(
        self,
        parent: ttk.Frame,
    ) -> ttk.Treeview:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        tree = ttk.Treeview(
            parent,
            show="headings",
        )

        vertical_scroll = ttk.Scrollbar(
            parent,
            orient="vertical",
            command=tree.yview,
        )

        horizontal_scroll = ttk.Scrollbar(
            parent,
            orient="horizontal",
            command=tree.xview,
        )

        tree.configure(
            yscrollcommand=vertical_scroll.set,
            xscrollcommand=horizontal_scroll.set,
        )

        tree.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        vertical_scroll.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        horizontal_scroll.grid(
            row=1,
            column=0,
            sticky="ew",
        )

        return tree

    def select_output_file(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select Modern AutoCrit Output",
            filetypes=[
                ("Excel files", "*.xlsx"),
                ("All files", "*.*"),
            ],
        )

        if filename:
            self.output_file_var.set(filename)
            self.refresh()

    def refresh(self) -> None:
        output_path = self._resolve_output_path()

        if output_path is None:
            messagebox.showwarning(
                "No Output File",
                "Run the pipeline or select an output workbook.",
            )
            return

        if not output_path.exists():
            messagebox.showerror(
                "Missing Output File",
                f"The selected file does not exist:\n{output_path}",
            )
            return

        self.output_file_var.set(
            str(output_path)
        )

        try:
            output_df = preview_output(
                output_excel=output_path,
                n=50,
            )
        except TypeError:
            # Compatibility with earlier preview_output signature.
            output_df = preview_output(
                output_path,
                n=50,
            )
        except Exception as exc:
            messagebox.showerror(
                "Preview Error",
                str(exc),
            )
            return

        self._load_dataframe_into_tree(
            output_df,
            self.output_tree,
        )

        unmapped_path = self._get_unmapped_path(
            output_path
        )

        if unmapped_path.exists():
            try:
                unmapped_df = pd.read_excel(
                    unmapped_path
                ).head(100)
            except Exception as exc:
                self.status_var.set(
                    f"Output loaded, but unmapped terms failed: {exc}"
                )
                unmapped_df = pd.DataFrame()
        else:
            unmapped_df = pd.DataFrame()

        self._load_dataframe_into_tree(
            unmapped_df,
            self.unmapped_tree,
        )

        self._refresh_summary_from_state(
            output_df
        )

        self.status_var.set(
            f"Loaded {len(output_df)} preview row(s) from {output_path.name}."
        )

    def _resolve_output_path(
        self,
    ) -> Path | None:
        manual_path = self.output_file_var.get().strip()

        if manual_path:
            return Path(manual_path)

        if self.state.current_output_file is not None:
            return Path(
                self.state.current_output_file
            )

        return None

    @staticmethod
    def _get_unmapped_path(
        output_path: Path,
    ) -> Path:
        return output_path.with_name(
            output_path.stem
            + "_unmapped_terms.xlsx"
        )

    def _refresh_summary_from_state(
        self,
        output_df: pd.DataFrame,
    ) -> None:
        summary = self.state.last_summary

        if summary is not None:
            self.trials_var.set(
                str(summary.trials_processed)
            )
            self.extracted_var.set(
                str(summary.criteria_extracted)
            )
            self.normalized_var.set(
                str(
                    summary.criteria_after_normalization
                )
            )
            self.deduplicated_var.set(
                str(
                    summary.criteria_after_deduplication
                )
            )
            self.cost_var.set(
                f"${summary.total_cost_usd:.4f}"
            )
            return

        # Fallback for manually loaded files.
        if "trial_id" in output_df.columns:
            self.trials_var.set(
                str(
                    output_df["trial_id"]
                    .nunique()
                )
            )
        else:
            self.trials_var.set("—")

        row_count = len(output_df)

        self.extracted_var.set(str(row_count))
        self.normalized_var.set(str(row_count))
        self.deduplicated_var.set(str(row_count))
        self.cost_var.set("—")

    def _load_dataframe_into_tree(
        self,
        df: pd.DataFrame,
        tree: ttk.Treeview,
    ) -> None:
        for item in tree.get_children():
            tree.delete(item)

        tree["columns"] = []

        if df is None or df.empty:
            return

        columns = [
            str(column)
            for column in df.columns
        ]

        tree["columns"] = columns

        for column in columns:
            tree.heading(
                column,
                text=column,
            )
            tree.column(
                column,
                width=150,
                anchor="w",
                stretch=True,
            )

        for row in df.itertuples(
            index=False,
            name=None,
        ):
            values = []

            for value in row:
                if pd.isna(value):
                    values.append("")
                else:
                    values.append(str(value))

            tree.insert(
                "",
                "end",
                values=values,
            )

    def open_output_file(self) -> None:
        output_path = self._resolve_output_path()

        if output_path is None:
            messagebox.showwarning(
                "No Output File",
                "No output file is available.",
            )
            return

        self._open_path(output_path)

    def open_unmapped_file(self) -> None:
        output_path = self._resolve_output_path()

        if output_path is None:
            messagebox.showwarning(
                "No Output File",
                "No output file is available.",
            )
            return

        unmapped_path = self._get_unmapped_path(
            output_path
        )

        if not unmapped_path.exists():
            messagebox.showwarning(
                "No Unmapped File",
                "No unmapped-terms workbook was found.",
            )
            return

        self._open_path(unmapped_path)

    @staticmethod
    def _open_path(path: Path) -> None:
        if not path.exists():
            messagebox.showerror(
                "Missing File",
                f"The file does not exist:\n{path}",
            )
            return

        try:
            if sys.platform == "darwin":
                subprocess.run(
                    ["open", str(path)],
                    check=True,
                )
            elif os.name == "nt":
                os.startfile(str(path))
            else:
                subprocess.run(
                    ["xdg-open", str(path)],
                    check=True,
                )
        except Exception as exc:
            messagebox.showerror(
                "Open File Error",
                str(exc),
            )