from __future__ import annotations

from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from backend.api.app_state import create_app_state
from frontend.views.run_view import RunView
from frontend.views.settings_view import SettingsView
from frontend.views.analytics_view import AnalyticsView
from frontend.components.dictionary_editor import DictionaryEditor


class ModernAutoCritApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.state = create_app_state()

        self.setWindowTitle("Modern AutoCrit")
        self.resize(1100, 750)

        tabs = QTabWidget()

        tabs.addTab(RunView(self.state), "Run")
        tabs.addTab(SettingsView(self.state), "Settings")
        tabs.addTab(DictionaryEditor(), "Dictionary")
        tabs.addTab(AnalyticsView(self.state), "Analytics")

        self.setCentralWidget(tabs)


def launch_app():
    app = QApplication([])

    window = ModernAutoCritApp()
    window.show()

    app.exec()