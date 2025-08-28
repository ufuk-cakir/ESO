import datetime
import requests
import json
import logging
import os
import subprocess
import platform
import sys
import threading
import time
import tkinter as tk
import webbrowser
import tkinter.font as tkFont
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

from eso.eso import ESO
from eso.utils.settings import Config

# Create a custom logger
logger = logging.getLogger(__name__)


# TODO maybe add checkbox to keep logs for each run, then generate folder called "runs" and store logs with timestamp there

from PyQt5.QtCore import QObject, pyqtSignal

class ESOProgressHandler(QObject):
    main_progress_max_changed = pyqtSignal(int)
    main_progress_value_changed = pyqtSignal(int)
    training_progress_max_changed = pyqtSignal(int)
    training_progress_value_changed = pyqtSignal(int)

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

class StdoutRedirector:
    def __init__(self, logger, log_level=logging.ERROR):
        self.logger = logger
        self.log_level = log_level
        self.buffer = ""
        self.error_shown = False

    def write(self, message):
        self.buffer += message
        if message.endswith("\n"):
            self.logger.log(self.log_level, self.buffer.rstrip())
            self.buffer = ""
            if self.log_level == logging.ERROR and not self.error_shown:
                messagebox.showerror("Error", "Check console log for more information.")
                self.error_shown = True

    def flush(self):
        pass


class TextHandler(logging.Handler):
    """Class to redirect logger output to a tkinter Text widget"""

    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        # Get the log entry and format it using the formatter set in the logger
        log_entry = self.format(record)
        self.text_widget.config(state="normal")  # Enable editing
        self.text_widget.insert(tk.END, log_entry + "\n")
        self.text_widget.see(tk.END)  # Scroll to the end
        self.text_widget.config(state="disabled")  # Disable editing
        self.text_widget.update_idletasks()


class ToolTip:
    """Class to create tooltips for a given widget

    Attributes:
        widget: The widget to bind the tooltip to
        text: The text to display in the tooltip
        FONT_SIZE_HOVER: The font size of the tooltip
        BORDER_WIDTH_HOVER: The border width of the tooltip

    Methods:
        showtip: Displays the tooltip. Called when the mouse enters the widget
        hidetip: Hides the tooltip. Called when the mouse leaves the widget
    """

    def __init__(self, widget, text, FONT_SIZE_HOVER=12, BORDER_WIDTH_HOVER=1):
        self.BORDER_WIDTH_HOVER = BORDER_WIDTH_HOVER
        self.FONT_SIZE_HOVER = FONT_SIZE_HOVER
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.text = text

        # Bind events Enter and Leave to the widget
        # Show the tooltip when the mouse enters the widget and hide it when mouse leaves the widget
        self.widget.bind("<Enter>", self.showtip)
        self.widget.bind("<Leave>", self.hidetip)

    def showtip(self, event=None):
        """Displays the tooltip"""

        # Get the relative coordinates of the cursor within the widget
        x, y, _, _ = self.widget.bbox("insert")

        # Get absolute coordinates of the widget and add some padding
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        # Create a toplevel window
        tw = tk.Toplevel(self.widget)
        self.tipwindow = tw

        # Remove any window decorations like border, title bar, etc.
        tw.wm_overrideredirect(1)

        # Set the position of the tooltip window
        tw.wm_geometry(f"+{x}+{y}")

        # Create a label inside the tooltip window with the text
        label = tk.Label(
            tw,
            text=self.text,
            relief="solid",
            borderwidth=self.BORDER_WIDTH_HOVER,
            padx=10,
            pady=5,
            font=("tahoma", f"{self.FONT_SIZE_HOVER}", "normal"),
        )

        # Pack the label into the tooltip window
        label.pack(ipadx=1)

    def hidetip(self, event=None):
        if self.tipwindow:
            self.tipwindow.destroy()


class App(tk.Tk):
    WIDTH = 650
    HEIGHT = 750
    CONSOLE_WIDHT = 650
    CONSOLE_HEIHT = 700

    def __init__(self):
        super().__init__()
        os.makedirs("logs/tensorboard", exist_ok=True)
        sys.stderr = StdoutRedirector(logger, logging.ERROR)
        sys.stdout = StdoutRedirector(logger, logging.INFO)

        self.title("Evolutionary Spectrogram Optimization")
        self.set_basic_geometry()
        self.protocol("WM_DELETE_WINDOW", self.close_window)
        self.resizable(False, False)

        menubar = tk.Menu(self)
        self.config(menu=menubar)
        # Hack!!! my font is not nice on the notebook frame
        if sys.platform != "linux":
            font_size = int(self.winfo_height() * 0.07)
            default_font = tkFont.nametofont("TkDefaultFont")
            default_font.configure(size=font_size)
            self.option_add("*Font", default_font)

        view_menu = tk.Menu(menubar, tearoff=0)
        load_menu = tk.Menu(menubar, tearoff=0)
        clear_menu = tk.Menu(menubar, tearoff=0)
        show_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Load", menu=load_menu)
        menubar.add_cascade(label="View", menu=view_menu)
        menubar.add_cascade(label="Clear", menu=clear_menu)
        menubar.add_cascade(label="Show", menu=show_menu)
        view_menu.add_command(label="Show Tensorboard...", command=self.show_training)
        view_menu.add_command(
            label="Show Console Log...", command=self.open_console_log
        )

        load_menu.add_command(label="Load Settings...", command=self.load_settings_file)
        clear_menu.add_command(
            label="Clear Tensorboard Runs...", command=self.clear_tensorboard_logs
        )
        show_menu.add_command(label="Show log folder...", command=self.show_log_folder)
        show_menu.add_command(
            label="Show settings folder...", command=self.show_settings_folder
        )
        self.tensorboard_log_dir = "logs/tensorboard"
        self.log_path = "logs/console.log"
        self.log_folder = "logs"
        self.results_path = "results"
        self.settings_path = "settings"
        os.makedirs(self.results_path, exist_ok=True)
        # get absolute path of results folder
        self.results_path = os.path.abspath(self.results_path)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(pady=10, expand=True)

        # Load the icon image and resize it
        icon_image = Image.open("img/info.png")
        resized_icon = icon_image.resize((20, 20), Image.LANCZOS)
        self.info_icon = ImageTk.PhotoImage(resized_icon, master=self)

        # Load the search icon image and resize it
        search_icon_image = Image.open("img/search_icon.png")
        resized_search_icon = search_icon_image.resize((20, 20), Image.LANCZOS)
        self.search_icon = ImageTk.PhotoImage(resized_search_icon, master=self)

        app_icon = Image.open("img/e_icon.png")
        # Source: <a target="_blank" href="https://icons8.com/icon/6mHOVAGdtvAI/e">E</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>
        # resized_app_icon = app_icon.resize((20, 20), Image.LANCZOS)
        self.app_icon = ImageTk.PhotoImage(app_icon, master=self)
        self.iconphoto(False, self.app_icon)
        self.wm_iconphoto(False, self.app_icon)

        # Variable to store the selected file path
        self.file_path = tk.StringVar()
        self.create_basic_frame()
        # Create and populate Advanced frame (to the right of Basic frame)
        style = ttk.Style(self)
        style.configure("lefttab.TNotebook", tabposition="wn")

        self.settings_notebook = ttk.Notebook(
            self.settings_frame, style="lefttab.TNotebook"
        )
        self.settings_notebook.pack(pady=1, expand=True)

        self.config = Config()
        self.settings = self.load_settings("settings/default_settings.json")
        self.default_settings = self.settings.copy()
        self._generate_types_dict()
        self.variables = {}
        self.create_settings_frame()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        # Add file handler to save logs to file
        file_handler = logging.FileHandler(self.log_path, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        # Update the file handler's level based on the checkbox value
        self.logger = logger
        self.update_file_handler_logging()

        self.console_open = False
        self.clear_console_log()
        self.current_epoch = 0

        # Check if settings file exists

        if os.path.exists("settings/settings.json"):
            self.load_settings_file("settings/settings.json")

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

    def _open_folder(self, path):
        system = platform.system()
        # Get absolute path
        path = os.path.abspath(path)
        if os.path.exists(path):
            try:
                if system == "Windows":
                    path = path.replace("/", "\\")
                    os.system(f"start {path}")
                elif system == "Darwin":
                    subprocess.Popen(["open", path])
                else:
                    subprocess.Popen(["xdg-open", path])
            except Exception:
                messagebox.showerror("Error", "Could not open Folder Path")
        else:
            messagebox.showerror("Error", "Folder path does not exist!")

    def show_settings_folder(self):
        self._open_folder(self.settings_path)

    def show_log_folder(self):
        self._open_folder(self.log_folder)

    def clear_tensorboard_logs(self):
        """Clears the tensorboard logs

        Clears the tensorboard logs by deleting the logs/tensorboard folder and recreating it.
        """
        # Ask for confirmation in a thread, so the GUI doesn't freeze
        self.confirmation_thread = threading.Thread(
            target=self._clear_tensorboard_logs_threaded
        )
        self.confirmation_thread.start()

    def _clear_tensorboard_logs_threaded(self):
        # Create message box to ask for confirmation
        answer = messagebox.askyesno(
            "Clear Tensorboard Logs",
            "Are you sure you want to clear the Tensorboard logs?",
        )
        if answer:
            # Check if algorithm is running
            if hasattr(self, "thread"):
                if self.thread.is_alive():
                    # Stop the algorithm
                    messagebox.showinfo(
                        "Algorithm Running",
                        "Please stop the algorithm before clearing the Tensorboard logs",
                    )
                    return
            # Delete the logs/tensorboard folder
            os.system("rm -rf logs/tensorboard")
            # Recreate the logs/tensorboard folder
            os.makedirs("logs/tensorboard", exist_ok=True)
            messagebox.showinfo(
                "Tensorboard Logs Cleared", "Tensorboard logs cleared successfully"
            )
        else:
            return

    def load_settings_file(self, path=None):
        if path is None:
            # Open file dialog to select settings file
            settings_path = filedialog.askopenfilename(
                initialdir="settings",
                title="Select file",
                filetypes=(("JSON files", "*.json"),),
            )
        else:
            settings_path = path
        if settings_path == "":
            return
        self.settings = self.load_settings(settings_path)
        self.update_settings()
        # Print
        self.logger.info(f"Loaded settings from {settings_path}")
        # show message box
        if path is None:
            messagebox.showinfo(
                "Settings Loaded", f"Loaded settings from {settings_path}"
            )

    def update_settings(self):
        # Update the settings in the GUI based on the loaded settings
        for settings_type, settings in self.settings.items():
            for key, data in settings.items():
                value = data

                if self.types[settings_type][key] == bool:
                    self.variables[settings_type][key].set(value)
                elif self.types[settings_type][key] == str:
                    if key == "species_folder":
                        self.file_path.set(value)
                        # self.data_path_entry.config(textvariable=value)
                    else:
                        self.variables[settings_type][key].set(str(value))
                elif self.types[settings_type][key] == int:
                    self.variables[settings_type][key].set(value)
                elif self.types[settings_type][key] == float:
                    self.variables[settings_type][key].set(value)
        # Print message to console log

    def create_basic_frame(self):
        # Create the Basic frame
        self.basic_frame = ttk.Frame(
            self.notebook, width=self.WIDTH, height=self.HEIGHT
        )
        self.settings_frame = ttk.Frame(
            self.notebook, width=self.WIDTH, height=self.HEIGHT
        )

        # Add widgets to select file path and run the algorithm
        self.data_path_label = ttk.Label(self.basic_frame, text="Select Data Path:")
        self.data_path_label.grid(row=0, column=0, pady=10, padx=10)

        self.data_path_entry = ttk.Entry(
            self.basic_frame, textvariable=self.file_path, state="readonly"
        )
        self.data_path_entry.grid(row=0, column=1, pady=10, padx=10, sticky=tk.E + tk.W)

        self.browse_button = ttk.Button(
            self.basic_frame, image=self.search_icon, command=self.browse_file
        )
        self.browse_button.grid(row=0, column=2, pady=10, padx=10)
        self.run_button = ttk.Button(
            self.basic_frame, text="Run", command=self.run_algorithm
        )
        self.run_button.grid(row=1, column=0, columnspan=3, pady=10)

        self.stop_button = ttk.Button(
            self.basic_frame, text="Stop", command=self.set_stop_flag
        )

        self.stop_button.grid(row=2, column=0, pady=2, columnspan=3)

        # Create the progress bar using the custom style
        self.progress_bar_label = ttk.Label(self.basic_frame, text="Progress:")
        self.progress_bar_label.grid(row=3, column=0, pady=0)
        self.progress_bar = ttk.Progressbar(
            self.basic_frame, orient=tk.HORIZONTAL, length=450, mode="determinate"
        )
        self.progress_bar.grid(row=4, column=0, columnspan=7, pady=0)
        self.notebook.add(self.basic_frame, text="Basic")

        # Add progressbar for population training
        self.progress_bar_label_training = ttk.Label(
            self.basic_frame, text="Training Progress:"
        )
        self.progress_bar_label_training.grid(row=5, column=0, pady=0)
        self.progress_bar_training = ttk.Progressbar(
            self.basic_frame, orient=tk.HORIZONTAL, length=450, mode="determinate"
        )
        self.progress_bar_training.grid(row=6, column=0, columnspan=7, pady=0)

        # Create frame to hold a matplotlib figure to display the spectrogram
        self.spectrogram_frame = ttk.Frame(
            self.basic_frame,
            width=self.SPECTROGRAM_WIDHT,
            height=self.SPECTROGRAM_HEIGHT,
        )
        self.spectrogram_frame.grid(row=7, column=0, columnspan=7, pady=10)
        self.spectrogram_frame.grid_propagate(False)
        self.spectrogram_frame.grid_rowconfigure(0, weight=1)
        self.spectrogram_frame.grid_columnconfigure(0, weight=1)

        # Create a label to display the spectrogram
        self.spectrogram_label = ttk.Label(self.spectrogram_frame)
        self.spectrogram_label.grid(row=0, column=0, sticky=tk.N + tk.S + tk.E + tk.W)
        self.spectrogram_label.grid_propagate(False)
        self.spectrogram_label.grid_rowconfigure(0, weight=1)
        self.spectrogram_label.grid_columnconfigure(0, weight=1)

        # Create a empty figure to display in the spectrogram label
        self.fig = plt.figure(figsize=(4.5, 4.5))
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self.ax.set_title("Optimized Bands")
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.spectrogram_label)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def update_spec_image(self):
        # load the image
        img = plt.imread(os.path.join(self.results_path, "current_best_chromosome.png"))
        # Update the image in the figure
        self.ax.imshow(img)
        # Update the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def check_if_epoch_increased(self):
        """Checks if the current epoch has increased"""
        if self.current_epoch == self.progress_bar["value"]:
            False
        else:
            self.current_epoch = self.progress_bar["value"]
            return True

    def create_settings_frame(self):
        self.notebook.add(self.settings_frame, text="Advanced")
        for settings_name in self.settings.keys():
            frame = ttk.Frame(self.settings_notebook)
            setattr(self, f"{settings_name}_frame", frame)
        """
        self.notebook.add(self.settings_frame, text="Advanced")
        self.algorithm_frame = ttk.Frame(self.settings_notebook)
        self.preprocessing_frame = ttk.Frame(self.settings_notebook)
        self.gene_frame = ttk.Frame(self.settings_notebook)
        self.chromosome_frame = ttk.Frame(self.settings_notebook)
        self.training_frame = ttk.Frame(self.settings_notebook)

        frames = [self.algorithm_frame, self.preprocessing_frame,
                  self.gene_frame, self.chromosome_frame, self.training_frame]
        keys = ["algorithm", "genetic_operator", "selection_operator", "data", "preprocess"]
        """
        frames = []
        keys = []
        for frame, settings_type in zip(
            self.settings_notebook.winfo_children(), self.settings.keys()
        ):
            # Loop through all frames of the settings notebook and create the widgets
            frames.append(frame)
            keys.append(settings_type)
            self.variables[settings_type] = {}
            for idx, (key, data) in enumerate(self.settings[settings_type].items()):
                # Loop through all settings for one type (algorithm, preprocessing, etc.)
                # Create widgets based on the type of the setting
                if key == "species_folder":
                    continue
                value = data["value"]
                description = data["description"]
                label = ttk.Label(frame, text=key.capitalize().replace("_", " ") + ":")
                label.grid(row=idx, column=0, sticky=tk.W, padx=10, pady=3)

                if self.types[settings_type][key] == bool:
                    var = tk.BooleanVar(value=value)
                    # Store the variable for later access
                    self.variables[settings_type][key] = var
                    checkbutton = ttk.Checkbutton(frame, variable=var)
                    checkbutton.grid(row=idx, column=1, sticky=tk.W, padx=10, pady=3)
                    info_label = tk.Label(frame, image=self.info_icon)
                    info_label.grid(row=idx, column=2, padx=5, pady=3)
                    ToolTip(info_label, description)
                elif self.types[settings_type][key] == str:
                    var = tk.StringVar(value=str(value))
                    # Store the variable for later access
                    self.variables[settings_type][key] = var
                    entry = ttk.Entry(frame, width=10, textvariable=var)
                    entry.grid(row=idx, column=1, pady=3)

                    info_label = tk.Label(frame, image=self.info_icon)
                    info_label.grid(row=idx, column=2, padx=5, pady=3)
                    ToolTip(info_label, description)

                elif self.types[settings_type][key] == int:
                    var = tk.IntVar(value=value)
                    self.variables[settings_type][key] = var
                    entry = ttk.Entry(frame, width=10, textvariable=var)
                    entry.grid(row=idx, column=1, pady=3)

                    info_label = tk.Label(frame, image=self.info_icon)
                    info_label.grid(row=idx, column=2, padx=5, pady=3)
                    ToolTip(info_label, description)

                elif self.types[settings_type][key] == float:
                    var = tk.DoubleVar(value=value)
                    self.variables[settings_type][key] = var
                    entry = ttk.Entry(frame, width=10, textvariable=var)
                    entry.grid(row=idx, column=1, pady=3)

                    info_label = tk.Label(frame, image=self.info_icon)
                    info_label.grid(row=idx, column=2, padx=5, pady=3)
                    ToolTip(info_label, description)

            self.debug_frame = ttk.Frame(self.settings_frame)

        # Checkbox wheter to open log file after run
        self.open_log_var = tk.BooleanVar(value=True)
        self.check_open_log = ttk.Checkbutton(
            self.debug_frame,
            text="Open console log after run",
            variable=self.open_log_var,
        )
        self.check_open_log.grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        info_label = tk.Label(self.debug_frame, image=self.info_icon)
        info_label.grid(row=0, column=1, padx=5, pady=3)
        ToolTip(info_label, "Opens the console log after the algorithm has finished")

        # Checkbox for log saving
        self.save_to_file_var = tk.BooleanVar(value=True)
        self.check_save_log = ttk.Checkbutton(
            self.debug_frame, text="Save logs to file", variable=self.save_to_file_var
        )
        self.check_save_log.grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        info_label = tk.Label(self.debug_frame, image=self.info_icon)
        info_label.grid(row=1, column=1, padx=5, pady=3)
        ToolTip(info_label, "Enables saving logs to file")

        # Checkbox wheter to enable Debug Mode
        self.debug_mode_var = tk.BooleanVar(value=True)
        self.check_debug_mode = ttk.Checkbutton(
            self.debug_frame, text="Debug Mode", variable=self.debug_mode_var
        )
        self.check_debug_mode.grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        info_label = tk.Label(self.debug_frame, image=self.info_icon)
        info_label.grid(row=2, column=1, padx=5, pady=3)
        ToolTip(
            info_label,
            "Enables debug mode, which will log debug messages to the console log",
        )
        frames.append(self.debug_frame)
        keys.append("debug")

        # Put image of fitness function equation in training frame
        fitess_image = Image.open("img/fitness.png")
        # fitess_image.resize((self.WIDTH, self.HEIGHT), Image.LANCZOS)
        resized_fitess_image = fitess_image
        self.fitness_image = ImageTk.PhotoImage(resized_fitess_image, master=self)
        self.fitness_image_label = tk.Label(
            self.chromosome_frame, image=self.fitness_image
        )
        self.fitness_image_label.grid(row=9, column=0, columnspan=4, padx=10, pady=5)

        for frame, settings_type in zip(frames, keys):
            frame.pack(fill=tk.BOTH, expand=True)
            text = settings_type.capitalize()
            # get rid of everything that comes after _
            text = text.split("_")[0]
            self.settings_notebook.add(frame, text=text)

    def estimate_time_left(self):
        self.time_estimate_thread = threading.Thread(
            target=self._estimate_time_left_threaded
        )
        # Close the thread when the window is closed
        self.time_estimate_thread.daemon = True
        self.time_estimate_thread.start()

    def _estimate_time_left_threaded(self):
        """Estimates the time left until the algorithm finishes

        Estimates the time left until the algorithm finishes based on the current epoch, the total number of epochs and the start time of the algorithm.
        """
        # TODO change this, so it doesn't use recursion
        # maybe use while loop
        if self.stop_event.is_set():
            self.reset_progress_bars()
            return
        if self.progress_bar["value"] == 0:
            self.current_epoch = 0
            self.progress_bar_label.config(
                text=f"Progress: Running... time left: Calculating..."
            )
            time.sleep(5)
            self._estimate_time_left_threaded()
        else:
            self.logger.debug("Estimating time...")
            if not self.check_if_epoch_increased():
                time.sleep(5)
                self._estimate_time_left_threaded()
            else:
                self.current_epoch = self.progress_bar["value"]
            current_epoch = self.progress_bar["value"]
            total_epoch = self.variables["algorithm"]["max_generations"].get()
            start_time = self.start_time
            # time spent per epoch
            time_per_epoch = (
                (datetime.datetime.now() - start_time) / current_epoch
                if current_epoch != 0
                else 0
            )
            epochs_left = total_epoch - current_epoch
            time_left = time_per_epoch * epochs_left
            # Format the time left
            time_left = str(time_left).split(".")[0]
            # set the text of the progressbar label to "Running... time left: ..."
            self.progress_bar_label.config(
                text=f"Progress: Running... time left: {time_left}"
            )

            # Update the progressbar label every 5 seconds
            time.sleep(5)
            # self.after(100, self.estimate_time_left)
            self.update_spec_image()
            self._estimate_time_left_threaded()

    def close_window(self):
        """Close window"""
        # W
        # Stop the algorithm if its running
        if hasattr(self, "thread"):
            if self.thread.is_alive():
                self.set_stop_flag()
        # Stop Tensorboard if its running
        self.stop_tensorboard()
        # Save specific settings?

        # Close the window
        self.destroy()

    def close_console_log(self):
        """Closes the console log window

        Called when the console log window is closed. Sets the console_open flag to False.
        """
        self.closed_console_log = True
        # Hide the window
        self.console_window.withdraw()

    def open_console_log(self):
        # Check if console log is already open
        if not self.console_open:  # Check if the console log frame is closed
            self.create_console_frame()  # If it's closed, create and open the console log frame
        else:
            self.console_window.deiconify()  # If it's open but hidden, show it again
            self.console_closed = False

    def create_console_frame(self):
        """Opens the console log window

        Opens the console log window if it is not already open.
        """
        self.console_window = tk.Toplevel(self)
        self.console_window.title("Console Log")
        self.console_window.geometry(f"{self.CONSOLE_WIDHT}x{self.CONSOLE_HEIHT}")
        self.console_window.resizable(True, True)
        # Make the display fullscreen
        self.console_display = tk.Text(
            self.console_window, height=40, width=45, state="disabled"
        )
        self.console_display.pack(fill=tk.BOTH, expand=True)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        text_handler = TextHandler(self.console_display)
        text_handler.setFormatter(formatter)
        self.logger.addHandler(text_handler)

        # Copy existing logs to the console log
        with open(self.log_path, "r") as log_file:
            log_content = log_file.read()
            self.console_display.config(state="normal")
            self.console_display.insert(tk.END, log_content)
            self.console_display.config(state="disabled")
        # Set flag if window is closed
        self.console_window.protocol("WM_DELETE_WINDOW", self.close_console_log)
        self.console_open = True

    def show_training(self):
        """Starts Tensorboard and opens the browser window

        Opens Tensorboard in a separate thread, so the GUI doesn't freeze while Tensorboard is running.
        """
        self.logger.info("Starting Tensorboard....")
        self.tensorboard_thread = threading.Thread(target=self._show_training_threaded)
        self.tensorboard_thread.start()

    def _wait_for_tensorboard(self, port=6006, timeout=60):
        """Wait until Tensorboard is accessible"""
        end_time = time.time() + timeout
        url = f"http://localhost:{port}/"
        while time.time() < end_time:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    return True
            except requests.ConnectionError:
                pass
            time.sleep(1)
        return False

    def _show_training_threaded(self):
        """Starts Tensorboard and opens the browser window

        This is a separate thread, so the GUI doesn't freeze while Tensorboard is running.
        """
        # Open browser window with tensorboard
        # Clear tensorboard logs
        # os.system("rm -rf logs/tensorboard/*")
        messagebox.showinfo("Tensorboard", "Launching Tensorboard, may take a while..")
        try:
            subprocess.Popen(["tensorboard", "--logdir", "logs/tensorboard"])
            # TODO Wait until tensorboard is started
            if self._wait_for_tensorboard():
                webbrowser.open("http://localhost:6006/")
                self.logger.info("Tensorboard started!!")
            else:
                messagebox.showerror("Error", "Tensorboard took too long to start.")
        except Exception as e:
            self.logger.error(e)
            messagebox.showerror("Error", f"Could not start Tensorboard: {e}")

    def set_file_handler_level(self, level):
        """Sets the file handler logging level to the given level"""
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.setLevel(level)

    def toggle_debug_mode(self):
        """Toggles debug mode based on the checkbox value"""
        if self.debug_mode_var.get():
            self.logger.setLevel(logging.DEBUG)
            self.set_file_handler_level(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
            self.set_file_handler_level(logging.INFO)

    def update_file_handler_logging(self):
        """Updates the file handler logging level based on the checkbox value"""
        # Enable/Disable the file handler based on the checkbox value
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                if self.save_to_file_var.get():
                    handler.setLevel(logging.INFO)
                else:
                    # Effectively disabling it
                    handler.setLevel(logging.CRITICAL)

    def reset_progress_bars(self):
        self.progress_bar_label.config(text="Progress: Stopped!")
        # Reset the progress bar
        self.progress_bar["value"] = 0
        self.progress_bar.update()
        self.progress_bar_training["value"] = 0
        self.progress_bar_training.update()
        self.current_epoch = 0

    def set_stop_flag(self):
        """Sets the stop flag to stop the algorithm

        This function is called when the Stop button is clicked. Sets the stop flag to stop the thread running the algorithm.
        """
        self.logger.info("Stopping algorithm... Waiting for current epoch to finish...")
        self.running = False
        self.stop_event.set()
        # set the text of the progressbar label to "Running...
        self.reset_progress_bars()

    def load_settings(self, file_name):
        """Loads the settings from the given file"""
        with open(file_name, "r") as f:
            return json.load(f)

    def set_basic_geometry(self):
        # Get geometry of the screen
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        # Calculate the x and y coordinates to center the window
        # Set a ratio to make the window smaller than the screen
        height = screen_height * 0.9
        width = screen_width * 0.45
        self.SPECTROGRAM_WIDHT = int(width * 0.7)
        self.SPECTROGRAM_HEIGHT = int(height / 2)
        self.WIDHT = width
        self.HEIGHT = height
        self.geometry(f"{int(width)}x{int(height)}")

    def stop_tensorboard(self):
        """Stops Tensorboard

        Stops Tensorboard by killing the process. This is necessary, because Tensorboard doesn't stop automatically.
        This function is called when the window is closed.
        """
        try:
            subprocess.run(["pkill", "-f", "tensorboard"])
        except Exception as e:
            self.logger.error(f"Error stopping Tensorboard: {e}")

    def browse_file(self):
        """Opens a file dialog to select a file and stores the path in the file_path variable"""
        this_directory = os.getcwd()
        file = filedialog.askdirectory(
            title="Select a Folder", initialdir=this_directory
        )
        if file:
            self.file_path.set(file)

    def clear_console_log(self):
        """Clears the console log

        Called when the Run button is clicked. Clears the console log and the log file.
        """
        with open(self.log_path, "w"):
            pass

    def save_settings(self):
        """Saves the settings to settings.json

        Saves the settings to settings.json. Called when the Run button is clicked.
        Loop through all variables and convert them to the correct data type defined in src/eso/utils/_settings_types.py.
        Saves the settings to settings/settings.json.
        """
        self.variables["data"]["species_folder"] = self.file_path

        settings = {}
        for category, variables in self.variables.items():
            settings[category] = {}
            for key, var in variables.items():
                # TODO change this, so its not so hacky
                # The problem is that the Entry widget returns a string, but we need to store the correct data type
                variable = var.get()
                # Convert the variable to the correct type
                variable = self.types[category][key](variable)
                settings[category][key] = variable
        self.logger.info("Saving settings to settings.json:", self.variables)

        with open("settings/settings.json", "w") as f:
            json.dump(settings, f, indent=4)

    def run_algorithm(self):
        """Creates a thread and runs the algorithm

        This function is called when the Run button is clicked. Creates a thread and runs the algorithm in it.
        """
        # Check if there is already an algorithm running
        if hasattr(self, "thread"):
            if self.thread.is_alive():
                messagebox.showerror("Error", "Algorithm is already running")
                return
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run_algorithm_threaded)
        self.thread.start()
        sys.stderr.error_shown = False
        sys.stdout.error_shown = False
        # will This run after thread is finished?

    def _run_algorithm_threaded(self):
        """Runs the algorithm with the selected settings

        This function is called when the Run button is clicked. Clears the console log, updates the file handler logging level,
        toggles debug mode, checks if a file is selected, initializes the algorithm and runs it.
        """

        self.clear_console_log()
        self.update_file_handler_logging()
        self.toggle_debug_mode()

        # set the text of the progressbar label to "Running...
        self.progress_bar_label.config(text="Progress: Running...")
        population_file_path = None
        # Check if Population File already exists
        if os.path.exists(os.path.abspath(os.path.join("results", "population.pkl"))):
            answer = messagebox.askyesno(
                "Population file found!",
                "Found already existing population File! Do you want to continue training?",
            )
            if answer:
                population_file_path = os.path.abspath(
                    os.path.join("results", "population.pkl")
                )
        # Set starting time
        self.start_time = datetime.datetime.now()
        # Check if a file is selected
        if not self.file_path.get():
            messagebox.showerror("Error", "Please select a file")
            return

        self.estimate_time_left()

        # open console log window
        if self.open_log_var.get():
            self.open_console_log()

        # Set up initial log message
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        init_message = f"Algorithm run at {timestamp} on file: {self.file_path.get()}"
        self.logger.info(init_message)

        # Save settings to settings.json
        self.save_settings()

        # TODO add a check if data type is correct and if file exists

        # Run the algorithm and catch any exceptions
        self.eso = ESO(
            settings="settings/settings.json",
            stop_event=self.stop_event,
            logger=self.logger,
            population_file_path=population_file_path,
            results_path=self.results_path,
            tensorboard_log_dir=self.tensorboard_log_dir,
            progress_bar=self.progress_bar,
            progress_bar_training=self.progress_bar_training,
        )

        self.eso.run()
        self.eso.save()


if __name__ == "__main__":
    app = App()
    app.mainloop()
