from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from backend.api.app_state import AppState
from backend.api.routes import save_settings


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


class SettingsView(QWidget):
    def __init__(self, state: AppState):
        super().__init__()

        self.state = state
        settings = self.state.settings
        llm = settings.llm

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h2>Settings</h2>"))

        form = QFormLayout()

        self.provider = QComboBox()
        self.provider.addItems(list(MODEL_OPTIONS.keys()))
        self.provider.setCurrentText(llm.provider)

        self.api_key = QLineEdit(llm.api_key)
        self.api_key.setEchoMode(QLineEdit.Password)

        self.model = QComboBox()
        self.model.setEditable(True)

        self.base_url = QLineEdit(llm.base_url or "")
        self.base_url.setPlaceholderText(
            "Optional, e.g. http://localhost:11434 for Ollama"
        )

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(llm.temperature)

        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(256, 32000)
        self.max_tokens.setValue(llm.max_output_tokens)

        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(50, 2000)
        self.chunk_size.setValue(settings.extraction.chunk_size)

        self.overlap = QSpinBox()
        self.overlap.setRange(0, 1000)
        self.overlap.setValue(settings.extraction.overlap)

        self.enable_normalization = QCheckBox()
        self.enable_normalization.setChecked(
            settings.extraction.enable_normalization
        )

        self.enable_dedup = QCheckBox()
        self.enable_dedup.setChecked(
            settings.extraction.enable_semantic_deduplication
        )

        self.warning_threshold = QDoubleSpinBox()
        self.warning_threshold.setRange(0.0, 10000.0)
        self.warning_threshold.setDecimals(2)
        self.warning_threshold.setValue(
            settings.cost.warning_threshold_usd
        )

        form.addRow("Provider", self.provider)
        form.addRow("API Key", self.api_key)
        form.addRow("Model", self.model)
        form.addRow("Base URL", self.base_url)
        form.addRow("Temperature", self.temperature)
        form.addRow("Max Output Tokens", self.max_tokens)
        form.addRow("Chunk Size", self.chunk_size)
        form.addRow("Overlap", self.overlap)
        form.addRow("Enable Normalization", self.enable_normalization)
        form.addRow(
            "Enable Semantic Deduplication",
            self.enable_dedup,
        )
        form.addRow(
            "Cost Warning Threshold ($)",
            self.warning_threshold,
        )

        layout.addLayout(form)

        self.status = QLabel("")
        layout.addWidget(self.status)

        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save)
        layout.addWidget(save_button)

        self.provider.currentTextChanged.connect(
            self.update_model_options
        )

        self.update_model_options(llm.provider)
        self.model.setCurrentText(llm.model)

    def update_model_options(self, provider: str) -> None:
        current_model = self.model.currentText()

        self.model.blockSignals(True)
        self.model.clear()
        self.model.addItems(MODEL_OPTIONS.get(provider, []))

        if current_model:
            self.model.setCurrentText(current_model)

        self.model.blockSignals(False)

        is_ollama = provider == "ollama"

        self.api_key.setEnabled(not is_ollama)
        self.base_url.setEnabled(is_ollama)

        if is_ollama and not self.base_url.text().strip():
            self.base_url.setText("http://localhost:11434")

    def save(self) -> None:
        settings = self.state.settings

        settings.llm.provider = self.provider.currentText().strip()
        settings.llm.api_key = self.api_key.text().strip()
        settings.llm.model = self.model.currentText().strip()
        settings.llm.base_url = (
            self.base_url.text().strip() or None
        )
        settings.llm.temperature = self.temperature.value()
        settings.llm.max_output_tokens = self.max_tokens.value()

        settings.extraction.chunk_size = self.chunk_size.value()
        settings.extraction.overlap = self.overlap.value()
        settings.extraction.enable_normalization = (
            self.enable_normalization.isChecked()
        )
        settings.extraction.enable_semantic_deduplication = (
            self.enable_dedup.isChecked()
        )

        settings.cost.warning_threshold_usd = (
            self.warning_threshold.value()
        )

        save_settings(self.state)

        self.status.setText("Settings saved.")