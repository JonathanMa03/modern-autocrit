import os
from PySide6.QtCore import QLibraryInfo

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.path(
    QLibraryInfo.PluginsPath
)

from frontend.app import launch_app


def main():
    launch_app()


if __name__ == "__main__":
    main()