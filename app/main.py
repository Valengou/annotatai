import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor

from app.ui.main_window import MainWindow


def apply_dark_theme(app: QApplication):
    app.setStyle("Fusion")
    palette = QPalette()
    dark = QColor(30, 30, 30)
    darker = QColor(20, 20, 20)
    mid = QColor(50, 50, 50)
    lighter = QColor(70, 70, 70)
    highlight = QColor(42, 130, 218)
    text = QColor(210, 210, 210)
    bright_text = QColor(255, 255, 255)

    palette.setColor(QPalette.Window, dark)
    palette.setColor(QPalette.WindowText, text)
    palette.setColor(QPalette.Base, darker)
    palette.setColor(QPalette.AlternateBase, dark)
    palette.setColor(QPalette.ToolTipBase, dark)
    palette.setColor(QPalette.ToolTipText, text)
    palette.setColor(QPalette.Text, text)
    palette.setColor(QPalette.Button, mid)
    palette.setColor(QPalette.ButtonText, text)
    palette.setColor(QPalette.BrightText, bright_text)
    palette.setColor(QPalette.Link, highlight)
    palette.setColor(QPalette.Highlight, highlight)
    palette.setColor(QPalette.HighlightedText, bright_text)
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, lighter)
    palette.setColor(QPalette.Disabled, QPalette.WindowText, lighter)
    palette.setColor(QPalette.Disabled, QPalette.Text, lighter)
    app.setPalette(palette)

    app.setStyleSheet("""
        QToolBar { border: none; padding: 4px; spacing: 4px; }
        QToolButton { padding: 4px 8px; border-radius: 4px; }
        QToolButton:hover { background: #3a3a3a; }
        QToolButton:pressed { background: #2a6fb5; }
        QTabWidget::pane { border: 1px solid #333; }
        QTabBar::tab { padding: 6px 14px; background: #2a2a2a; border: 1px solid #333;
                       border-bottom: none; border-radius: 4px 4px 0 0; }
        QTabBar::tab:selected { background: #1e1e1e; border-color: #555; }
        QTabBar::tab:hover { background: #363636; }
        QPushButton { padding: 5px 12px; border-radius: 4px; background: #3a3a3a;
                      border: 1px solid #555; }
        QPushButton:hover { background: #484848; }
        QPushButton:pressed { background: #2a6fb5; }
        QPushButton:disabled { color: #555; background: #2a2a2a; }
        QLineEdit { background: #2a2a2a; border: 1px solid #444; border-radius: 3px;
                    padding: 4px; }
        QComboBox { background: #2a2a2a; border: 1px solid #444; border-radius: 3px;
                    padding: 4px; }
        QScrollBar:vertical { width: 10px; background: #1e1e1e; }
        QScrollBar::handle:vertical { background: #555; border-radius: 5px; min-height: 20px; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QTreeWidget { border: none; }
        QListWidget { border: none; }
        QSplitter::handle { background: #333; }
        QProgressBar { border: 1px solid #444; border-radius: 3px; text-align: center; }
        QProgressBar::chunk { background: #2a6fb5; }
        QMenuBar { background: #1e1e1e; border-bottom: 1px solid #333; }
        QMenuBar::item:selected { background: #3a3a3a; }
        QMenu { background: #2a2a2a; border: 1px solid #444; }
        QMenu::item:selected { background: #2a6fb5; }
        QStatusBar { border-top: 1px solid #333; }
    """)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AnnotatAI")
    app.setOrganizationName("DroneAI")
    apply_dark_theme(app)
    window = MainWindow()
    window.show()

    # Auto-open project if path passed as argument
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        project_path = Path(args[0])
        if project_path.exists():
            window.open_project(project_path)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
