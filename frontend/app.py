from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from backend.api.app_state import create_app_state
from frontend.views.run_view import RunView
from frontend.views.settings_view import SettingsView
from frontend.views.dictionary_view import DictionaryView
from frontend.views.analytics_view import AnalyticsView


class ModernAutoCritApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.state = create_app_state()

        self.title("Modern AutoCrit")
        self.geometry("1100x750")
        self.minsize(900, 600)

        self._configure_style()
        self._build_ui()

    def _configure_style(self) -> None:
        style = ttk.Style(self)

        if "aqua" in style.theme_names():
            style.theme_use("aqua")
        elif "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure(
            "Title.TLabel",
            font=("Helvetica", 18, "bold"),
        )

        style.configure(
            "Section.TLabel",
            font=("Helvetica", 12, "bold"),
        )

    def _build_ui(self) -> None:
        container = ttk.Frame(
            self,
            padding=10,
        )
        container.pack(
            fill="both",
            expand=True,
        )

        title = ttk.Label(
            container,
            text="Modern AutoCrit",
            style="Title.TLabel",
        )
        title.pack(
            anchor="w",
            pady=(0, 10),
        )

        self.notebook = ttk.Notebook(container)
        self.notebook.pack(
            fill="both",
            expand=True,
        )

        self.run_view = RunView(
            self.notebook,
            self.state,
        )

        self.settings_view = SettingsView(
            self.notebook,
            self.state,
        )

        self.dictionary_view = DictionaryView(
            self.notebook,
        )

        self.analytics_view = AnalyticsView(
            self.notebook,
            self.state,
        )

        self.notebook.add(
            self.run_view,
            text="Run",
        )
        self.notebook.add(
            self.settings_view,
            text="Settings",
        )
        self.notebook.add(
            self.dictionary_view,
            text="Dictionary",
        )
        self.notebook.add(
            self.analytics_view,
            text="Analytics",
        )


def launch_app() -> None:
    app = ModernAutoCritApp()
    app.mainloop()