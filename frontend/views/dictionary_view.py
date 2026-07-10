from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from backend.api.routes import (
    add_dictionary_mapping,
    load_dictionary,
    save_dictionary,
)


class DictionaryView(ttk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
    ) -> None:
        super().__init__(parent, padding=15)

        self.dictionary_type_var = tk.StringVar(value="attribute")
        self.search_var = tk.StringVar()
        self.raw_term_var = tk.StringVar()
        self.canonical_term_var = tk.StringVar()
        self.status_var = tk.StringVar(value="")

        self.current_mapping: dict[str, str] = {}

        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        ttk.Label(
            self,
            text="Dictionary Editor",
            style="Section.TLabel",
        ).grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 15),
        )

        top_frame = ttk.Frame(self)
        top_frame.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(0, 10),
        )
        top_frame.columnconfigure(3, weight=1)

        ttk.Label(
            top_frame,
            text="Dictionary",
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 8),
        )

        self.dictionary_combo = ttk.Combobox(
            top_frame,
            textvariable=self.dictionary_type_var,
            values=[
                "attribute",
                "entity",
                "disease",
            ],
            state="readonly",
            width=18,
        )
        self.dictionary_combo.grid(
            row=0,
            column=1,
            sticky="w",
        )
        self.dictionary_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.refresh(),
        )

        ttk.Label(
            top_frame,
            text="Search",
        ).grid(
            row=0,
            column=2,
            sticky="w",
            padx=(20, 8),
        )

        self.search_entry = ttk.Entry(
            top_frame,
            textvariable=self.search_var,
        )
        self.search_entry.grid(
            row=0,
            column=3,
            sticky="ew",
        )
        self.search_var.trace_add(
            "write",
            lambda *_args: self.populate_table(),
        )

        ttk.Button(
            top_frame,
            text="Refresh",
            command=self.refresh,
        ).grid(
            row=0,
            column=4,
            padx=(10, 0),
        )

        table_frame = ttk.Frame(self)
        table_frame.grid(
            row=3,
            column=0,
            sticky="nsew",
        )
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        columns = (
            "raw_term",
            "canonical_term",
        )

        self.table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            selectmode="browse",
        )

        self.table.heading(
            "raw_term",
            text="Raw Term",
        )
        self.table.heading(
            "canonical_term",
            text="Canonical Term",
        )

        self.table.column(
            "raw_term",
            width=350,
            anchor="w",
        )
        self.table.column(
            "canonical_term",
            width=350,
            anchor="w",
        )

        vertical_scroll = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.table.yview,
        )
        horizontal_scroll = ttk.Scrollbar(
            table_frame,
            orient="horizontal",
            command=self.table.xview,
        )

        self.table.configure(
            yscrollcommand=vertical_scroll.set,
            xscrollcommand=horizontal_scroll.set,
        )

        self.table.grid(
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

        self.table.bind(
            "<<TreeviewSelect>>",
            self._on_row_selected,
        )

        editor_frame = ttk.LabelFrame(
            self,
            text="Add or Update Mapping",
            padding=12,
        )
        editor_frame.grid(
            row=4,
            column=0,
            sticky="ew",
            pady=(12, 0),
        )
        editor_frame.columnconfigure(1, weight=1)

        ttk.Label(
            editor_frame,
            text="Raw Term",
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=5,
        )

        ttk.Entry(
            editor_frame,
            textvariable=self.raw_term_var,
        ).grid(
            row=0,
            column=1,
            sticky="ew",
            pady=5,
        )

        ttk.Label(
            editor_frame,
            text="Canonical Term",
        ).grid(
            row=1,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=5,
        )

        ttk.Entry(
            editor_frame,
            textvariable=self.canonical_term_var,
        ).grid(
            row=1,
            column=1,
            sticky="ew",
            pady=5,
        )

        controls = ttk.Frame(editor_frame)
        controls.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(10, 0),
        )

        ttk.Button(
            controls,
            text="Add / Update Mapping",
            command=self.add_or_update_mapping,
        ).pack(side="left")

        ttk.Button(
            controls,
            text="Delete Selected",
            command=self.delete_selected,
        ).pack(
            side="left",
            padx=(10, 0),
        )

        ttk.Button(
            controls,
            text="Clear Fields",
            command=self.clear_fields,
        ).pack(
            side="left",
            padx=(10, 0),
        )

        ttk.Label(
            controls,
            textvariable=self.status_var,
        ).pack(
            side="left",
            padx=(15, 0),
        )

    def refresh(self) -> None:
        dictionary_type = self.dictionary_type_var.get()

        try:
            self.current_mapping = load_dictionary(
                dictionary_type
            )
        except Exception as exc:
            messagebox.showerror(
                "Dictionary Error",
                str(exc),
            )
            return

        self.populate_table()
        self.status_var.set(
            f"Loaded {len(self.current_mapping)} mappings."
        )

    def populate_table(self) -> None:
        for item in self.table.get_children():
            self.table.delete(item)

        search_text = self.search_var.get().strip().lower()

        rows = sorted(
            self.current_mapping.items(),
            key=lambda pair: pair[0],
        )

        for raw_term, canonical_term in rows:
            if search_text:
                combined = (
                    f"{raw_term} {canonical_term}"
                ).lower()

                if search_text not in combined:
                    continue

            self.table.insert(
                "",
                "end",
                values=(
                    raw_term,
                    canonical_term,
                ),
            )

    def _on_row_selected(self, _event=None) -> None:
        selected = self.table.selection()

        if not selected:
            return

        values = self.table.item(
            selected[0],
            "values",
        )

        if len(values) < 2:
            return

        self.raw_term_var.set(values[0])
        self.canonical_term_var.set(values[1])

    def add_or_update_mapping(self) -> None:
        raw_term = self.raw_term_var.get().strip()
        canonical_term = (
            self.canonical_term_var.get().strip()
        )

        if not raw_term or not canonical_term:
            messagebox.showwarning(
                "Missing Mapping",
                "Both raw and canonical terms are required.",
            )
            return

        dictionary_type = self.dictionary_type_var.get()

        try:
            self.current_mapping = add_dictionary_mapping(
                dictionary_type=dictionary_type,
                raw_term=raw_term,
                canonical_term=canonical_term,
            )
        except Exception as exc:
            messagebox.showerror(
                "Dictionary Error",
                str(exc),
            )
            return

        self.populate_table()
        self.clear_fields()
        self.status_var.set("Mapping saved.")

    def delete_selected(self) -> None:
        selected = self.table.selection()

        if not selected:
            messagebox.showwarning(
                "No Selection",
                "Select a mapping to delete.",
            )
            return

        values = self.table.item(
            selected[0],
            "values",
        )

        if not values:
            return

        raw_term = str(values[0])

        confirmed = messagebox.askyesno(
            "Delete Mapping",
            f"Delete the mapping for '{raw_term}'?",
        )

        if not confirmed:
            return

        self.current_mapping.pop(raw_term, None)

        try:
            save_dictionary(
                self.dictionary_type_var.get(),
                self.current_mapping,
            )
        except Exception as exc:
            messagebox.showerror(
                "Dictionary Error",
                str(exc),
            )
            return

        self.refresh()
        self.clear_fields()
        self.status_var.set("Mapping deleted.")

    def clear_fields(self) -> None:
        self.raw_term_var.set("")
        self.canonical_term_var.set("")

        for item in self.table.selection():
            self.table.selection_remove(item)