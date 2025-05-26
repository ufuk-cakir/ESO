import sys
import os
import platform
import subprocess
import threading
import time
import datetime
import json
import logging

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QProgressBar,
    QTextEdit,
    QFileDialog,
    QMessageBox,
    QTabWidget,
    QGroupBox,
    QAction,
    QMenuBar,
    QGraphicsDropShadowEffect,
    QScrollArea,
)
from PyQt5.QtGui import QIcon, QPixmap, QTextCursor, QColor, QFont
from PyQt5.QtCore import pyqtSignal, QObject, Qt, QSize

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtCore import QObject, pyqtSignal

from eso import ESO
from eso.utils.settings import Config


class ESOProgressHandler(QObject):
    main_progress_max_changed = pyqtSignal(int)
    main_progress_value_changed = pyqtSignal(int)
    training_progress_max_changed = pyqtSignal(int)
    training_progress_value_changed = pyqtSignal(int)
    best_chromosome_image_updated = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

    def set_main_max(self, value):
        self.main_progress_max_changed.emit(value)

    def set_main_value(self, value):
        self.main_progress_value_changed.emit(value)

    def set_training_max(self, value):
        self.training_progress_max_changed.emit(value)

    def set_training_value(self, value):
        self.training_progress_value_changed.emit(value)

    def notify_best_chromosome_image_updated(self, image_path: str):
        self.best_chromosome_image_updated.emit(image_path)


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class QtStream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass


class StyledEsoApp(QMainWindow):
    progress_updated = pyqtSignal(int)
    training_progress_updated = pyqtSignal(int)
    spectrogram_path_updated = pyqtSignal(str)
    algorithm_finished = pyqtSignal(bool)

    APP_WIDTH = 800
    APP_HEIGHT = 850

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Evolutionary Spectrogram Optimization")
        self.setGeometry(100, 100, self.APP_WIDTH, self.APP_HEIGHT)
        font = QFont("Inter")
        font.setPointSize(10)
        QApplication.setFont(font)

        try:
            self.setWindowIcon(QIcon("img/e_icon.png"))
        except Exception as e:
            logger.warning(f"Could not load app icon: {e}")

        self.current_settings_path = "settings/default_settings.json"
        self.settings_data_cache = {}
        self.results_path = "results_qt_styled"
        self.log_folder = "logs_qt_styled"
        self.tensorboard_log_dir = os.path.join(self.log_folder, "tensorboard")
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)

        self.worker_thread = None
        self.stop_event = threading.Event()
        self.tensorboard_process = None

        self.settings_widgets = {}

        self._create_actions()
        self._create_menu_bar()

        self.progress_handler = ESOProgressHandler(self)
        self._init_ui()

        self.progress_handler.main_progress_max_changed.connect(
            self.main_progress_bar.setMaximum
        )
        self.progress_handler.main_progress_value_changed.connect(
            self.main_progress_bar.setValue
        )
        self.progress_handler.training_progress_max_changed.connect(
            self.training_progress_bar.setMaximum
        )
        self.progress_handler.training_progress_value_changed.connect(
            self.training_progress_bar.setValue
        )
        self.progress_handler.best_chromosome_image_updated.connect(
            self.update_spectrogram_display
        )

        self.algorithm_finished.connect(self.on_algorithm_finished_ui)
        self.default_settings = {}
        self.types = {}
        self._load_and_prepare_default_settings(self.current_settings_path)

        self.load_settings(self.current_settings_path)

    def _load_and_prepare_default_settings(self, path):
        """Loads the default settings structure and generates the types dictionary."""
        try:
            with open(path, "r") as f:
                self.default_settings = json.load(f)
            self._generate_types_dict()
        except Exception as e:
            logger.error(
                f"Failed to load or prepare default settings from {path}: {e}",
                exc_info=True,
            )
            self.default_settings = {}
            self.types = {}

    def _generate_types_dict(self):
        """
        Generate a dictionary of Python types for casting settings values.
        This assumes self.default_settings contains the verbose format with "type" string.
        """
        self.types = {}
        if not self.default_settings:
            logger.warning("_generate_types_dict: self.default_settings is empty.")
            return

        for category, params in self.default_settings.items():
            self.types[category] = {}
            if isinstance(params, dict):
                for key, details in params.items():
                    if isinstance(details, dict) and "type" in details:
                        type_str = details["type"].lower()
                        if type_str == "bool":
                            self.types[category][key] = bool
                        elif type_str == "str":
                            self.types[category][key] = str
                        elif type_str == "int":
                            self.types[category][key] = int
                        elif type_str == "float":
                            self.types[category][key] = float
                        else:
                            logger.debug(
                                f"Unknown type string '{type_str}' for {category}.{key}"
                            )
        logger.debug(f"Generated self.types: {self.types}")

    def _create_shadow(self):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 30))
        return shadow

    def _create_actions(self):
        self.load_settings_action = QAction(
            QIcon.fromTheme("document-open", QIcon("img/folder-open.png")),
            "&Load Settings...",
            self,
        )
        self.load_settings_action.triggered.connect(self.load_settings_dialog)

        self.save_settings_action = QAction(
            QIcon.fromTheme("document-save", QIcon("img/save.png")),
            "&Save Settings",
            self,
        )
        self.save_settings_action.triggered.connect(self.save_settings)

        self.exit_action = QAction(
            QIcon.fromTheme("application-exit", QIcon("img/exit.png")), "E&xit", self
        )
        self.exit_action.triggered.connect(self.close)

        self.show_tensorboard_action = QAction(
            QIcon("img/tensorboard.png"), "&Show TensorBoard", self
        )  # Custom icon
        self.show_tensorboard_action.triggered.connect(self.show_tensorboard)

        self.clear_tensorboard_action = QAction(
            QIcon.fromTheme("edit-clear", QIcon("img/clear.png")),
            "&Clear TensorBoard Logs",
            self,
        )
        self.clear_tensorboard_action.triggered.connect(self.clear_tensorboard_logs)

        self.open_log_folder_action = QAction(
            QIcon("img/folder-log.png"), "Show Log Folder", self
        )
        self.open_log_folder_action.triggered.connect(
            lambda: self._open_folder_path(self.log_folder)
        )

        self.open_results_folder_action = QAction(
            QIcon("img/folder-results.png"), "Show Results Folder", self
        )
        self.open_results_folder_action.triggered.connect(
            lambda: self._open_folder_path(self.results_path)
        )

    def save_settings(self):
        """Gather current UI values and write them to settings/settings.json."""
        plain_settings = self.collect_settings_from_ui()
        try:
            os.makedirs("settings", exist_ok=True)
            with open("settings/settings.json", "w") as f:
                json.dump(plain_settings, f, indent=4)
            logger.info("Settings saved to settings/settings.json")
        except Exception as e:
            logger.error(f"Failed to save settings: {e}", exc_info=True)
            QMessageBox.critical(self, "Save Error", f"Could not save settings:\n{e}")

    def _create_menu_bar(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self.load_settings_action)
        file_menu.addAction(self.save_settings_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        view_menu = menu_bar.addMenu("&View")
        view_menu.addAction(self.show_tensorboard_action)
        view_menu.addAction(self.open_log_folder_action)
        view_menu.addAction(self.open_results_folder_action)

        tools_menu = menu_bar.addMenu("&Tools")
        tools_menu.addAction(self.clear_tensorboard_action)

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_app_layout = QVBoxLayout(central_widget)
        main_app_layout.setContentsMargins(15, 15, 15, 15)
        main_app_layout.setSpacing(15)

        self.control_group = QGroupBox("Controls")
        self.control_group.setObjectName("controlCard")
        self.control_group.setGraphicsEffect(self._create_shadow())
        control_layout = QVBoxLayout()
        control_layout.setSpacing(10)

        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Data Path:"))
        self.data_path_lineedit = QLineEdit()
        self.data_path_lineedit.setPlaceholderText(
            "Select folder containing audio data..."
        )
        self.data_path_lineedit.setReadOnly(True)
        path_layout.addWidget(self.data_path_lineedit, 1)
        self.browse_button = QPushButton(
            QIcon.fromTheme("folder-open", QIcon("img/folder-open.png")), "Browse..."
        )
        self.browse_button.setObjectName("browseButton")
        self.browse_button.setIconSize(QSize(16, 16))
        self.browse_button.clicked.connect(self.browse_data_folder)
        path_layout.addWidget(self.browse_button)
        control_layout.addLayout(path_layout)

        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        self.run_button = QPushButton(
            QIcon.fromTheme("media-playback-start", QIcon("img/run.png")),
            "Run Algorithm",
        )
        self.run_button.setIconSize(QSize(16, 16))
        self.run_button.clicked.connect(self.run_algorithm)
        button_layout.addWidget(self.run_button)

        self.stop_button = QPushButton(
            QIcon.fromTheme("media-playback-stop", QIcon("img/stop.png")),
            "Stop Algorithm",
        )
        self.stop_button.setIconSize(QSize(16, 16))
        self.stop_button.clicked.connect(self.stop_algorithm)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        control_layout.addLayout(button_layout)

        self.control_group.setLayout(control_layout)
        main_app_layout.addWidget(self.control_group)

        self.output_group = QGroupBox("Output & Progress")
        self.output_group.setObjectName("outputCard")
        self.output_group.setGraphicsEffect(self._create_shadow())
        output_layout = QVBoxLayout()
        output_layout.setSpacing(10)

        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.figure.patch.set_alpha(0.0)  # Transparent figure background
        self.spectrogram_canvas = FigureCanvas(self.figure)
        self.ax_spectrogram = self.figure.add_subplot(111)
        self.ax_spectrogram.set_title("Optimized Bands")
        self.ax_spectrogram.axis("off")
        self.ax_spectrogram.set_facecolor("white")  # Match card background
        self.figure.tight_layout(pad=0.5)  # Minimal padding
        output_layout.addWidget(
            self.spectrogram_canvas, 1
        )  # Canvas takes available vertical space

        progress_section_layout = QVBoxLayout()
        progress_section_layout.setSpacing(5)

        overall_prog_label = QLabel("Overall Progress:")
        overall_prog_label.setStyleSheet("font-weight:500; margin-top: 5px;")
        progress_section_layout.addWidget(overall_prog_label)
        self.main_progress_bar = QProgressBar()
        self.main_progress_bar.setTextVisible(True)  # Show percentage
        progress_section_layout.addWidget(self.main_progress_bar)

        training_prog_label = QLabel("Current Epoch Training:")
        training_prog_label.setStyleSheet("font-weight:500; margin-top: 5px;")
        progress_section_layout.addWidget(training_prog_label)
        self.training_progress_bar = QProgressBar()
        self.training_progress_bar.setTextVisible(True)
        progress_section_layout.addWidget(self.training_progress_bar)
        output_layout.addLayout(progress_section_layout)

        self.output_group.setLayout(output_layout)
        main_app_layout.addWidget(self.output_group, 1)  # Output card takes more space

        self.tab_widget = QTabWidget()
        self.tab_widget.setIconSize(QSize(16, 16))  # Icons for tabs

        self.settings_tab_container = QWidget()  # Container for QSS styling
        self.settings_tab_container.setObjectName("settingsTabContainer")
        settings_tab_outer_layout = QVBoxLayout(self.settings_tab_container)
        settings_tab_outer_layout.setContentsMargins(15, 15, 15, 15)

        self.settings_scroll_area = QScrollArea()  # Make settings scrollable
        self.settings_scroll_area.setWidgetResizable(True)
        self.settings_scroll_area.setStyleSheet(
            "QScrollArea { border: none; background-color:transparent; }"
        )  # Style scroll area

        self.settings_tab_content_widget = QWidget()  # Actual content for settings
        self.settings_layout = QVBoxLayout(
            self.settings_tab_content_widget
        )  # Layout for dynamic widgets
        self.settings_layout.setAlignment(Qt.AlignTop)
        self.settings_layout.setSpacing(12)

        self.settings_scroll_area.setWidget(self.settings_tab_content_widget)
        settings_tab_outer_layout.addWidget(self.settings_scroll_area)
        self.tab_widget.addTab(
            self.settings_tab_container, QIcon("img/settings.png"), "Settings"
        )

        self.console_tab_container = QWidget()
        self.console_tab_container.setObjectName("consoleTabContainer")
        console_tab_layout = QVBoxLayout(self.console_tab_container)
        console_tab_layout.setContentsMargins(
            15, 15, 15, 15
        )  # Padding inside the console "card"
        self.console_output_textedit = QTextEdit()
        self.console_output_textedit.setReadOnly(True)
        console_tab_layout.addWidget(self.console_output_textedit)
        self.tab_widget.addTab(
            self.console_tab_container, QIcon("img/console.png"), "Console Log"
        )

        main_app_layout.addWidget(self.tab_widget, 1)  # Tabs also take available space

        self._redirect_streams()

    def _redirect_streams(self):
        sys.stdout = QtStream()
        sys.stdout.textWritten.connect(self._append_to_console)
        sys.stderr = QtStream()
        sys.stderr.textWritten.connect(
            lambda text: self._append_to_console(
                f"<span style='color: #D92D20;'>{text}</span>"
            )
        )

    def _append_to_console(self, text):
        self.console_output_textedit.moveCursor(QTextCursor.End)
        processed_text = text.replace("\n", "<br/>").replace("\r", "")
        self.console_output_textedit.insertHtml(processed_text)
        self.console_output_textedit.ensureCursorVisible()

    def browse_data_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Data Folder", os.getcwd()
        )
        if folder_path:
            self.data_path_lineedit.setText(folder_path)
            logger.info(f"Data folder selected: {folder_path}")

    def load_settings_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Settings", "settings", "JSON files (*.json);;All files (*)"
        )
        if path:
            self.load_settings(path)

    def _generate_types_dict(self) -> None:
        """
        Generate a dictionary of types for the settings

        Returns
        -------
        None

        """

        self.types = {}
        for setting_type in self.default_settings:
            self.types[setting_type] = {}
            for key in self.default_settings[setting_type]:
                type_str = self.default_settings[setting_type][key]["type"]
                if type_str == "bool":
                    self.types[setting_type][key] = bool
                elif type_str == "str":
                    self.types[setting_type][key] = str
                elif type_str == "int":
                    self.types[setting_type][key] = int
                elif type_str == "float":
                    self.types[setting_type][key] = float
                else:
                    self.logger.debug(f"Unknown or incorrect input type: {type_str}")

    def load_settings(self, path):
        try:
            with open(path, "r") as f:
                settings_data_from_file = json.load(f)

            self.settings_data_cache = {}
            for category, default_params in self.default_settings.items():
                self.settings_data_cache[category] = {}
                loaded_params = settings_data_from_file.get(category, {})
                for key, default_details in default_params.items():
                    loaded_detail_or_value = loaded_params.get(key, default_details)

                    current_value = default_details.get("value")
                    if (
                        isinstance(loaded_detail_or_value, dict)
                        and "value" in loaded_detail_or_value
                    ):
                        current_value = loaded_detail_or_value["value"]
                    elif not isinstance(loaded_detail_or_value, dict):
                        current_value = loaded_detail_or_value

                    self.settings_data_cache[category][key] = {
                        "value": current_value,
                        "type": default_details.get("type", "str"),
                        "description": default_details.get("description", ""),
                    }

            self.current_settings_path = path
            self._populate_settings_ui(self.settings_data_cache)
            logger.info(
                f"Settings loaded from {path} and UI populated based on default structure."
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Settings",
                f"Could not load settings from {path}:\n{e}",
            )
            logger.error(f"Error loading settings: {e}", exc_info=True)

    def _populate_settings_ui(self, settings_data):
        for i in reversed(range(self.settings_layout.count())):
            widget_to_remove = self.settings_layout.itemAt(i).widget()
            if widget_to_remove:
                widget_to_remove.setParent(None)
                widget_to_remove.deleteLater()
        self.settings_widgets.clear()

        if not settings_data:
            self.settings_layout.addWidget(
                QLabel("No settings data loaded or settings file is empty.")
            )
            return

        for category, params in settings_data.items():
            if not isinstance(params, dict):
                continue

            category_group = QGroupBox(category.replace("_", " ").title())
            category_group.setStyleSheet(
                "QGroupBox { border: 1px solid #EAECF0; border-radius: 8px; margin-top: 10px; padding:10px; } "
                "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 0 5px 5px 5px; }"
            )
            category_layout = QVBoxLayout(category_group)
            category_layout.setSpacing(8)
            self.settings_widgets[category] = {}

            for key, details in params.items():
                if (
                    not isinstance(details, dict)
                    or "value" not in details
                    or "type" not in details
                ):
                    logger.warning(f"Skipping malformed setting: {category}/{key}")
                    continue

                value = details["value"]
                param_type = details["type"].lower()
                description = details.get("description", "")

                param_layout = QHBoxLayout()
                label_text = key.replace("_", " ").title()
                label = QLabel(f"{label_text}:")
                label.setFixedWidth(180)
                param_layout.addWidget(label)

                widget = None
                if param_type == "bool":
                    from PyQt5.QtWidgets import QCheckBox

                    widget = QCheckBox()
                    widget.setChecked(bool(value))
                elif param_type == "int":
                    from PyQt5.QtWidgets import QSpinBox

                    widget = QSpinBox()
                    widget.setRange(-2147483648, 2147483647)
                    widget.setValue(int(value))
                elif param_type == "float":
                    from PyQt5.QtWidgets import QDoubleSpinBox

                    widget = QDoubleSpinBox()
                    widget.setDecimals(3)  # Or more
                    widget.setRange(-1.0e38, 1.0e38)
                    widget.setValue(float(value))
                elif param_type == "str":
                    widget = QLineEdit(str(value))
                else:
                    widget = QLabel(f"Unsupported type: {param_type} ({value})")

                if widget:
                    widget.setToolTip(description)
                    param_layout.addWidget(widget, 1)  # Widget takes expanding space
                    self.settings_widgets[category][key] = widget

                category_layout.addLayout(param_layout)
            self.settings_layout.addWidget(category_group)
        self.settings_layout.addStretch(1)  # Push content up

    def collect_settings_from_ui(self):
        plain_settings = {}

        data_path_value = self.data_path_lineedit.text()
        if data_path_value:  # Only add if a path is selected
            plain_settings["data"] = plain_settings.get(
                "data", {}
            )  # Ensure 'data' category exists
            plain_settings["data"]["species_folder"] = data_path_value

        if not self.settings_data_cache:
            logger.warning(
                "settings_data_cache is empty, cannot fully collect settings from UI."
            )

        for category, params_details in self.settings_data_cache.items():
            if category not in plain_settings:
                plain_settings[category] = {}

            if category not in self.settings_widgets:
                logger.debug(
                    f"Category '{category}' from cache has no UI widgets, skipping for plain value extraction."
                )
                continue

            for key, details_dict in params_details.items():
                if key not in self.settings_widgets[category]:
                    if isinstance(details_dict, dict) and "value" in details_dict:
                        plain_settings[category][key] = details_dict["value"]
                    else:
                        logger.debug(
                            f"Key '{key}' in category '{category}' from cache not in UI and not verbose, skipping."
                        )
                    continue

                widget = self.settings_widgets[category][key]
                param_type = details_dict.get("type", "").lower()

                new_value = None
                if param_type == "bool":
                    from PyQt5.QtWidgets import QCheckBox

                    if isinstance(widget, QCheckBox):
                        new_value = widget.isChecked()
                elif param_type == "int":
                    from PyQt5.QtWidgets import QSpinBox

                    if isinstance(widget, QSpinBox):
                        new_value = widget.value()
                elif param_type == "float":
                    from PyQt5.QtWidgets import QDoubleSpinBox

                    if isinstance(widget, QDoubleSpinBox):
                        new_value = widget.value()
                elif param_type == "str":
                    if category == "data" and key == "species_folder":
                        new_value = self.data_path_lineedit.text()
                    elif isinstance(widget, QLineEdit):
                        new_value = widget.text()

                if new_value is not None:
                    if (
                        hasattr(self, "types")
                        and category in self.types
                        and key in self.types[category]
                    ):
                        try:
                            target_type = self.types[category][key]
                            plain_settings[category][key] = target_type(new_value)
                        except ValueError:
                            logger.warning(
                                f"Could not convert {new_value} to {target_type} for {category}.{key}"
                            )
                            plain_settings[category][key] = new_value  # Store as is
                    else:
                        plain_settings[category][key] = new_value
                elif category == "data" and key == "species_folder":
                    plain_settings[category][key] = data_path_value
                else:
                    logger.warning(
                        f"Could not get value for {category}.{key} of type {param_type}"
                    )

        logger.debug(
            f"Collected plain settings: {json.dumps(plain_settings, indent=2)}"
        )
        return plain_settings

    def eso_update_main_progress(self, current: int, total: int):
        pct = int(current / total * 100) if total else 0
        self.progress_updated.emit(pct)

    def eso_update_training_progress(self, current: int, total: int):
        pct = int(current / total * 100) if total else 0
        self.training_progress_updated.emit(pct)

    def eso_update_spectrogram(self, image_path: str):
        self.spectrogram_path_updated.emit(image_path)

    def run_algorithm(self):
        if not self.data_path_lineedit.text():
            QMessageBox.warning(self, "Input Missing", "Please select a data folder.")
            return

        if self.worker_thread and self.worker_thread.is_alive():
            QMessageBox.information(self, "Busy", "Algorithm is already running.")
            return

        current_ui_settings = self.collect_settings_from_ui()

        plain_ui_settings = self.collect_settings_from_ui()

        self.current_run_settings_path = os.path.abspath(
            "settings/settings.json"
        )  # Fixed path
        settings_dir = os.path.dirname(self.current_run_settings_path)
        os.makedirs(settings_dir, exist_ok=True)

        try:
            with open(self.current_run_settings_path, "w") as f:
                json.dump(plain_ui_settings, f, indent=4)
            logger.info(
                f"Settings for this run saved to {self.current_run_settings_path} (plain format)"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Settings Error", f"Could not save settings for the run: {e}"
            )
            logger.error(
                f"Failed to save {self.current_run_settings_path}: {e}", exc_info=True
            )
            return

        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stop_event.clear()

        logger.info(
            f"Starting ESO algorithm with data: {self.data_path_lineedit.text()}"
        )

        self.worker_thread = threading.Thread(
            target=self._algorithm_worker, args=(self.current_run_settings_path,)
        )
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def _algorithm_worker(self, settings_file_for_run):
        population_file_path = None
        potential_pop_file = os.path.join(self.results_path, "population.pkl")
        if os.path.exists(potential_pop_file):
            logger.info(
                "Found existing population.pkl. TODO: Prompt user via main thread."
            )
            population_file_path = potential_pop_file

        start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"ESO run started at {start_time_str} using settings: {settings_file_for_run}"
        )
        print(f"ESO run started at {start_time_str}")
        logger.debug(
            f"Using population file: {population_file_path if population_file_path else 'None'}"
        )
        logger.debug(f"Using settings file: {settings_file_for_run}")

        self.eso_instance = ESO(
            settings="settings/settings.json",
            stop_event=self.stop_event,
            logger=logger,
            population_file_path=population_file_path,
            results_path=self.results_path,
            tensorboard_log_dir=self.tensorboard_log_dir,
            progress_handler=self.progress_handler,
        )

        try:
            self.eso_instance = ESO(
                settings="settings/settings.json",
                stop_event=self.stop_event,
                logger=logger,
                population_file_path=population_file_path,
                results_path=self.results_path,
                tensorboard_log_dir=self.tensorboard_log_dir,
                progress_handler=self.progress_handler,
            )

            logger.info("ESO instance created. Starting run...")
            self.eso_instance.run()
            logger.info("ESO run completed. Saving results...")
            self.eso_instance.save()
            logger.info("ESO results saved.")
            self.algorithm_finished.emit(True)

        except Exception as e:
            logger.error(f"Error during ESO execution: {e}", exc_info=True)
            print(f"ERROR in ESO Algorithm: {e}")  # To GUI console
            self.algorithm_finished.emit(False)
        finally:
            logger.info(
                f"ESO worker thread finished for settings: {settings_file_for_run}"
            )

    def stop_algorithm(self):
        if self.worker_thread and self.worker_thread.is_alive():
            logger.info("Stop signal sent to algorithm.")
            print("GUI: Requesting algorithm stop...")
            self.stop_event.set()
            self.stop_button.setEnabled(False)
        else:
            logger.info("No algorithm running to stop.")

    def on_algorithm_finished_ui(self, completed):
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if completed:
            self.main_progress_bar.setValue(100)
            QMessageBox.information(
                self, "Finished", "Algorithm completed successfully."
            )
            logger.info("Algorithm finished successfully.")
        else:
            QMessageBox.warning(
                self,
                "Stopped / Error",
                "Algorithm was stopped or encountered an error.",
            )
            logger.info("Algorithm stopped or errored.")

    def update_spectrogram_display(self, image_path):
        try:
            image_path = os.path.abspath(image_path)
            if os.path.exists(image_path):
                img_data = plt.imread(image_path)
                self.ax_spectrogram.clear()
                self.ax_spectrogram.imshow(img_data)
                self.ax_spectrogram.set_title("Updated Spectrogram")
                self.ax_spectrogram.axis("off")
                self.figure.tight_layout(pad=0.5)
                self.spectrogram_canvas.draw_idle()
            else:
                logger.warning(f"Spectrogram image not found: {image_path}")
        except Exception as e:
            logger.error(f"Error updating spectrogram display: {e}")

    def show_tensorboard(self):
        if self.tensorboard_process and self.tensorboard_process.poll() is None:
            QMessageBox.information(
                self, "TensorBoard", "TensorBoard is already running."
            )
            import webbrowser

            webbrowser.open("http://localhost:6006")
            return
        try:
            logger.info(
                f"Launching TensorBoard with logdir: {self.tensorboard_log_dir}"
            )
            print(f"Launching TensorBoard with logdir: {self.tensorboard_log_dir}")
            self.tensorboard_process = subprocess.Popen(
                ["tensorboard", "--logdir", self.tensorboard_log_dir]
            )
            import webbrowser

            threading.Timer(
                2.0, lambda: webbrowser.open("http://localhost:6006")
            ).start()
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "TensorBoard command not found.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not start TensorBoard: {e}")

    def clear_tensorboard_logs(self):
        reply = QMessageBox.question(
            self,
            "Confirm Clear",
            f"Delete all in\n{self.tensorboard_log_dir}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            try:
                import shutil

                if os.path.exists(self.tensorboard_log_dir):
                    shutil.rmtree(self.tensorboard_log_dir)
                os.makedirs(self.tensorboard_log_dir, exist_ok=True)
                QMessageBox.information(self, "Success", "TensorBoard logs cleared.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to clear logs: {e}")

    def _open_folder_path(self, path):
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            QMessageBox.warning(self, "Not Found", f"Path does not exist: {abs_path}")
            return
        try:
            if platform.system() == "Windows":
                os.startfile(abs_path)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", abs_path])
            else:
                subprocess.Popen(["xdg-open", abs_path])
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Could not open path '{abs_path}': {e}"
            )

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.stop_event.set()
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=1)
            if self.tensorboard_process and self.tensorboard_process.poll() is None:
                self.tensorboard_process.terminate()
                try:
                    self.tensorboard_process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    self.tensorboard_process.kill()
            event.accept()
        else:
            event.ignore()


def apply_stylesheet(app: QApplication, path: str) -> None:
    with open(path, "r") as f:
        app.setStyleSheet(f.read())


if __name__ == "__main__":
    os.makedirs("settings", exist_ok=True)
    os.makedirs("img", exist_ok=True)  # For icons

    default_settings_file = "settings/default_settings.json"

    app = QApplication(sys.argv)
    apply_stylesheet(app, "styles/modern.qss")
    main_window = StyledEsoApp()
    main_window.show()
    sys.exit(app.exec_())
