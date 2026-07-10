from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from backend.api.app_state import AppState
from backend.api.routes import save_settings
from backend.schemas.settings_schema import LLMSettings
from backend.services.llm.factory import create_llm_provider


MODEL_OPTIONS = {
    "openai": [
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "gpt-4.1",
    ],
    "anthropic": [
        "claude-haiku-4-5",
        "claude-sonnet-4-5",
    ],
    "gemini": [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ],
    "ollama": [
        "llama3.2",
        "qwen2.5",
        "mistral",
    ],
}


class SettingsView(ttk.Frame):
    """
    Tkinter settings tab.

    Supports:
        - provider selection
        - model selection
        - API key or local base URL
        - generation settings
        - extraction settings
        - cost settings
        - provider/model availability testing
    """

    def __init__(
        self,
        parent: tk.Widget,
        state: AppState,
    ) -> None:
        super().__init__(parent, padding=15)

        self.state = state
        self.test_result_queue: queue.Queue[dict] = queue.Queue()
        self.test_thread: threading.Thread | None = None

        settings = self.state.settings
        llm = settings.llm

        self.provider_var = tk.StringVar(value=llm.provider)
        self.api_key_var = tk.StringVar(value=llm.api_key)
        self.model_var = tk.StringVar(value=llm.model)
        self.base_url_var = tk.StringVar(value=llm.base_url or "")
        self.temperature_var = tk.DoubleVar(value=llm.temperature)
        self.max_tokens_var = tk.IntVar(value=llm.max_output_tokens)

        self.chunk_size_var = tk.IntVar(
            value=settings.extraction.chunk_size
        )
        self.overlap_var = tk.IntVar(
            value=settings.extraction.overlap
        )
        self.normalization_var = tk.BooleanVar(
            value=settings.extraction.enable_normalization
        )
        self.semantic_dedup_var = tk.BooleanVar(
            value=settings.extraction.enable_semantic_deduplication
        )

        self.warning_threshold_var = tk.DoubleVar(
            value=settings.cost.warning_threshold_usd
        )

        self.status_message_var = tk.StringVar(value="")
        self.availability_var = tk.StringVar(value="Not tested")

        self._build_ui()
        self._update_provider_fields()

        self.after(100, self._poll_test_results)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)

        title = ttk.Label(
            self,
            text="Settings",
            style="Section.TLabel",
        )
        title.grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 15),
        )

        # -------------------------------------------------
        # LLM provider settings
        # -------------------------------------------------

        llm_frame = ttk.LabelFrame(
            self,
            text="Language Model Provider",
            padding=12,
        )
        llm_frame.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(0, 12),
        )
        llm_frame.columnconfigure(1, weight=1)

        ttk.Label(
            llm_frame,
            text="Provider",
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=5,
        )

        self.provider_combo = ttk.Combobox(
            llm_frame,
            textvariable=self.provider_var,
            values=list(MODEL_OPTIONS.keys()),
            state="readonly",
        )
        self.provider_combo.grid(
            row=0,
            column=1,
            sticky="ew",
            pady=5,
        )
        self.provider_combo.bind(
            "<<ComboboxSelected>>",
            self._on_provider_changed,
        )

        ttk.Label(
            llm_frame,
            text="Model",
        ).grid(
            row=1,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=5,
        )

        self.model_combo = ttk.Combobox(
            llm_frame,
            textvariable=self.model_var,
            values=MODEL_OPTIONS.get(
                self.provider_var.get(),
                [],
            ),
        )
        self.model_combo.grid(
            row=1,
            column=1,
            sticky="ew",
            pady=5,
        )

        ttk.Label(
            llm_frame,
            text="API Key",
        ).grid(
            row=2,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=5,
        )

        self.api_key_entry = ttk.Entry(
            llm_frame,
            textvariable=self.api_key_var,
            show="•",
        )
        self.api_key_entry.grid(
            row=2,
            column=1,
            sticky="ew",
            pady=5,
        )

        ttk.Label(
            llm_frame,
            text="Base URL",
        ).grid(
            row=3,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=5,
        )

        self.base_url_entry = ttk.Entry(
            llm_frame,
            textvariable=self.base_url_var,
        )
        self.base_url_entry.grid(
            row=3,
            column=1,
            sticky="ew",
            pady=5,
        )

        ttk.Label(
            llm_frame,
            text="Temperature",
        ).grid(
            row=4,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=5,
        )

        self.temperature_spinbox = ttk.Spinbox(
            llm_frame,
            from_=0.0,
            to=2.0,
            increment=0.1,
            textvariable=self.temperature_var,
            width=10,
        )
        self.temperature_spinbox.grid(
            row=4,
            column=1,
            sticky="w",
            pady=5,
        )

        ttk.Label(
            llm_frame,
            text="Max Output Tokens",
        ).grid(
            row=5,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=5,
        )

        self.max_tokens_spinbox = ttk.Spinbox(
            llm_frame,
            from_=256,
            to=32000,
            increment=256,
            textvariable=self.max_tokens_var,
            width=12,
        )
        self.max_tokens_spinbox.grid(
            row=5,
            column=1,
            sticky="w",
            pady=5,
        )

        # Availability test area
        test_frame = ttk.Frame(llm_frame)
        test_frame.grid(
            row=6,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(12, 0),
        )
        test_frame.columnconfigure(2, weight=1)

        self.test_button = ttk.Button(
            test_frame,
            text="Test Provider / Model",
            command=self.test_provider,
        )
        self.test_button.grid(
            row=0,
            column=0,
            sticky="w",
        )

        ttk.Label(
            test_frame,
            text="Status:",
        ).grid(
            row=0,
            column=1,
            sticky="w",
            padx=(15, 5),
        )

        self.availability_label = tk.Label(
            test_frame,
            textvariable=self.availability_var,
            fg="#666666",
            font=("Helvetica", 11, "bold"),
            anchor="w",
        )
        self.availability_label.grid(
            row=0,
            column=2,
            sticky="w",
        )

        # -------------------------------------------------
        # Extraction settings
        # -------------------------------------------------

        extraction_frame = ttk.LabelFrame(
            self,
            text="Extraction",
            padding=12,
        )
        extraction_frame.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(0, 12),
        )
        extraction_frame.columnconfigure(1, weight=1)

        ttk.Label(
            extraction_frame,
            text="Chunk Size",
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=5,
        )

        ttk.Spinbox(
            extraction_frame,
            from_=50,
            to=5000,
            increment=50,
            textvariable=self.chunk_size_var,
            width=12,
        ).grid(
            row=0,
            column=1,
            sticky="w",
            pady=5,
        )

        ttk.Label(
            extraction_frame,
            text="Chunk Overlap",
        ).grid(
            row=1,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=5,
        )

        ttk.Spinbox(
            extraction_frame,
            from_=0,
            to=2000,
            increment=10,
            textvariable=self.overlap_var,
            width=12,
        ).grid(
            row=1,
            column=1,
            sticky="w",
            pady=5,
        )

        ttk.Checkbutton(
            extraction_frame,
            text="Enable terminology normalization",
            variable=self.normalization_var,
        ).grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="w",
            pady=5,
        )

        ttk.Checkbutton(
            extraction_frame,
            text="Enable semantic deduplication",
            variable=self.semantic_dedup_var,
        ).grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="w",
            pady=5,
        )

        # -------------------------------------------------
        # Cost settings
        # -------------------------------------------------

        cost_frame = ttk.LabelFrame(
            self,
            text="Cost Monitoring",
            padding=12,
        )
        cost_frame.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(0, 12),
        )
        cost_frame.columnconfigure(1, weight=1)

        ttk.Label(
            cost_frame,
            text="Warning Threshold ($)",
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=5,
        )

        ttk.Spinbox(
            cost_frame,
            from_=0.0,
            to=10000.0,
            increment=1.0,
            textvariable=self.warning_threshold_var,
            width=12,
        ).grid(
            row=0,
            column=1,
            sticky="w",
            pady=5,
        )

        # -------------------------------------------------
        # Save controls
        # -------------------------------------------------

        controls = ttk.Frame(self)
        controls.grid(
            row=4,
            column=0,
            sticky="ew",
            pady=(5, 0),
        )

        ttk.Button(
            controls,
            text="Save Settings",
            command=self.save,
        ).pack(side="left")

        ttk.Label(
            controls,
            textvariable=self.status_message_var,
        ).pack(
            side="left",
            padx=(15, 0),
        )

    def _on_provider_changed(self, _event=None) -> None:
        provider = self.provider_var.get().strip().lower()

        available_models = MODEL_OPTIONS.get(provider, [])

        self.model_combo.configure(
            values=available_models,
        )

        current_model = self.model_var.get().strip()

        if (
            not current_model
            or current_model not in available_models
        ):
            if available_models:
                self.model_var.set(available_models[0])
            else:
                self.model_var.set("")

        self._update_provider_fields()
        self._set_availability_not_tested()

    def _update_provider_fields(self) -> None:
        provider = self.provider_var.get().strip().lower()

        if provider == "ollama":
            self.api_key_entry.configure(state="disabled")
            self.base_url_entry.configure(state="normal")

            if not self.base_url_var.get().strip():
                self.base_url_var.set(
                    "http://localhost:11434"
                )
        else:
            self.api_key_entry.configure(state="normal")
            self.base_url_entry.configure(state="disabled")

    def _set_availability_not_tested(self) -> None:
        self.availability_var.set("Not tested")
        self.availability_label.configure(
            fg="#666666",
        )

    def _build_test_settings(self) -> LLMSettings:
        return LLMSettings(
            provider=self.provider_var.get().strip(),
            api_key=self.api_key_var.get().strip(),
            model=self.model_var.get().strip(),
            base_url=(
                self.base_url_var.get().strip()
                or None
            ),
            temperature=0.0,
            max_output_tokens=20,
        )

    def test_provider(self) -> None:
        if self.test_thread and self.test_thread.is_alive():
            return

        provider = self.provider_var.get().strip()
        model = self.model_var.get().strip()

        if not provider:
            messagebox.showwarning(
                "Missing Provider",
                "Select an LLM provider.",
            )
            return

        if not model:
            messagebox.showwarning(
                "Missing Model",
                "Select or enter a model.",
            )
            return

        if (
            provider != "ollama"
            and not self.api_key_var.get().strip()
        ):
            messagebox.showwarning(
                "Missing API Key",
                "Enter an API key before testing the provider.",
            )
            return

        self.availability_var.set("Testing...")
        self.availability_label.configure(
            fg="#b36b00",
        )
        self.test_button.configure(
            state="disabled",
        )

        test_settings = self._build_test_settings()

        self.test_thread = threading.Thread(
            target=self._test_provider_worker,
            args=(test_settings,),
            daemon=True,
        )
        self.test_thread.start()

    def _test_provider_worker(
        self,
        test_settings: LLMSettings,
    ) -> None:
        try:
            provider = create_llm_provider(
                settings=test_settings,
                cost_monitor=None,
            )

            response = provider.generate(
                prompt=(
                    "Reply with exactly the single word: AVAILABLE"
                ),
                model=test_settings.model,
                temperature=0.0,
                max_output_tokens=20,
            )

            response_text = response.text.strip()

            if not response_text:
                raise RuntimeError(
                    "The provider returned an empty response."
                )

            self.test_result_queue.put(
                {
                    "available": True,
                    "message": (
                        f"Available — "
                        f"{test_settings.provider} / "
                        f"{test_settings.model}"
                    ),
                }
            )

        except Exception as exc:
            self.test_result_queue.put(
                {
                    "available": False,
                    "message": str(exc),
                }
            )

    def _poll_test_results(self) -> None:
        while True:
            try:
                result = self.test_result_queue.get_nowait()
            except queue.Empty:
                break

            self.test_button.configure(
                state="normal",
            )

            if result["available"]:
                self.availability_var.set(
                    result["message"]
                )
                self.availability_label.configure(
                    fg="#16833b",
                )
            else:
                self.availability_var.set(
                    "Not Available"
                )
                self.availability_label.configure(
                    fg="#b42318",
                )

                messagebox.showerror(
                    "Provider Test Failed",
                    result["message"],
                )

        self.after(
            100,
            self._poll_test_results,
        )

    def save(self) -> None:
        settings = self.state.settings

        settings.llm.provider = (
            self.provider_var.get().strip()
        )
        settings.llm.api_key = (
            self.api_key_var.get().strip()
        )
        settings.llm.model = (
            self.model_var.get().strip()
        )
        settings.llm.base_url = (
            self.base_url_var.get().strip()
            or None
        )
        settings.llm.temperature = (
            self.temperature_var.get()
        )
        settings.llm.max_output_tokens = (
            self.max_tokens_var.get()
        )

        settings.extraction.chunk_size = (
            self.chunk_size_var.get()
        )
        settings.extraction.overlap = (
            self.overlap_var.get()
        )
        settings.extraction.enable_normalization = (
            self.normalization_var.get()
        )
        settings.extraction.enable_semantic_deduplication = (
            self.semantic_dedup_var.get()
        )

        settings.cost.warning_threshold_usd = (
            self.warning_threshold_var.get()
        )

        try:
            save_settings(self.state)
        except Exception as exc:
            messagebox.showerror(
                "Settings Error",
                str(exc),
            )
            return

        self.status_message_var.set(
            "Settings saved."
        )