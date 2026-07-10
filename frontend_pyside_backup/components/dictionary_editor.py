from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QComboBox,
    QMessageBox,
)

from frontend.components.file_table import FileTable

from backend.api.routes import (
    load_dictionary,
    save_dictionary,
    add_dictionary_mapping,
)

import pandas as pd


class DictionaryEditor(QWidget):

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        # ------------------------
        # Dictionary selector
        # ------------------------

        top = QHBoxLayout()

        top.addWidget(QLabel("Dictionary"))

        self.dictionary = QComboBox()
        self.dictionary.addItems(
            [
                "attribute",
                "entity",
                "disease",
            ]
        )

        self.dictionary.currentTextChanged.connect(
            self.refresh
        )

        top.addWidget(self.dictionary)

        layout.addLayout(top)

        # ------------------------
        # Table
        # ------------------------

        self.table = FileTable()

        layout.addWidget(self.table)

        # ------------------------
        # Inputs
        # ------------------------

        self.raw_term = QLineEdit()
        self.raw_term.setPlaceholderText("Raw term")

        self.canonical_term = QLineEdit()
        self.canonical_term.setPlaceholderText("Canonical term")

        layout.addWidget(self.raw_term)
        layout.addWidget(self.canonical_term)

        # ------------------------
        # Buttons
        # ------------------------

        buttons = QHBoxLayout()

        add = QPushButton("Add Mapping")
        add.clicked.connect(self.add_mapping)

        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)

        save = QPushButton("Save")

        buttons.addWidget(add)
        buttons.addWidget(refresh)
        buttons.addWidget(save)

        layout.addLayout(buttons)

        self.refresh()

    def refresh(self):

        mapping = load_dictionary(
            self.dictionary.currentText()
        )

        df = pd.DataFrame(
            mapping.items(),
            columns=[
                "Raw Term",
                "Canonical Term",
            ],
        )

        self.table.load_dataframe(df)

    def add_mapping(self):

        raw = self.raw_term.text().strip()

        canonical = self.canonical_term.text().strip()

        if not raw or not canonical:

            QMessageBox.warning(
                self,
                "Missing Data",
                "Both fields are required.",
            )

            return

        add_dictionary_mapping(
            self.dictionary.currentText(),
            raw,
            canonical,
        )

        self.raw_term.clear()
        self.canonical_term.clear()

        self.refresh()