from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFormLayout,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
)

from backend.api.app_state import AppState
from backend.api.routes import save_settings


class SettingsView(QWidget):
    def __init__(self, state: AppState):
        super().__init__()

        self.state = state
        s = self.state.settings

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h2>Settings</h2>"))

        form = QFormLayout()

        self.api_key = QLineEdit(s.openai.api_key)
        self.api_key.setEchoMode(QLineEdit.Password)

        self.model = QLineEdit(s.openai.model)

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(s.openai.temperature)

        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(256, 32000)
        self.max_tokens.setValue(s.openai.max_output_tokens)

        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(50, 2000)
        self.chunk_size.setValue(s.extraction.chunk_size)

        self.overlap = QSpinBox()
        self.overlap.setRange(0, 1000)
        self.overlap.setValue(s.extraction.overlap)

        self.enable_normalization = QCheckBox()
        self.enable_normalization.setChecked(s.extraction.enable_normalization)

        self.enable_dedup = QCheckBox()
        self.enable_dedup.setChecked(s.extraction.enable_semantic_deduplication)

        self.warning_threshold = QDoubleSpinBox()
        self.warning_threshold.setRange(0.0, 10000.0)
        self.warning_threshold.setValue(s.cost.warning_threshold_usd)

        form.addRow("OpenAI API Key", self.api_key)
        form.addRow("Model", self.model)
        form.addRow("Temperature", self.temperature)
        form.addRow("Max Output Tokens", self.max_tokens)
        form.addRow("Chunk Size", self.chunk_size)
        form.addRow("Overlap", self.overlap)
        form.addRow("Enable Normalization", self.enable_normalization)
        form.addRow("Enable Semantic Deduplication", self.enable_dedup)
        form.addRow("Cost Warning Threshold ($)", self.warning_threshold)

        layout.addLayout(form)

        self.status = QLabel("")
        layout.addWidget(self.status)

        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save)
        layout.addWidget(save_button)

    def save(self):
        s = self.state.settings

        s.openai.api_key = self.api_key.text().strip()
        s.openai.model = self.model.text().strip()
        s.openai.temperature = self.temperature.value()
        s.openai.max_output_tokens = self.max_tokens.value()

        s.extraction.chunk_size = self.chunk_size.value()
        s.extraction.overlap = self.overlap.value()
        s.extraction.enable_normalization = self.enable_normalization.isChecked()
        s.extraction.enable_semantic_deduplication = self.enable_dedup.isChecked()

        s.cost.warning_threshold_usd = self.warning_threshold.value()

        save_settings(self.state)

        self.status.setText("Settings saved.")