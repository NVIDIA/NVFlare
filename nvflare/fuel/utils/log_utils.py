# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import json
import logging
import logging.config
import os
import re
from logging import Logger
from logging.handlers import RotatingFileHandler

from nvflare.apis.workspace import Workspace


class ANSIColor:
    GREY = "38"
    YELLOW = "33"
    RED = "31"
    BOLD_RED = "31;1"
    CYAN = "36"
    RESET = "0"


DEFAULT_LEVEL_COLORS = {
    "DEBUG": ANSIColor.GREY,
    "INFO": ANSIColor.GREY,
    "WARNING": ANSIColor.YELLOW,
    "ERROR": ANSIColor.RED,
    "CRITICAL": ANSIColor.BOLD_RED,
}


class BaseFormatter(logging.Formatter):
    def __init__(self, fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt=None, style="%"):
        """BaseFormatter is the default formatter for log records.

        Shortens logger %(name)s to the suffix. Full name can be accessed with %(fullName)s

        Args:
            fmt (str): format string which uses LogRecord attributes.
            datefmt (str): date/time format string. Defaults to '%Y-%m-%d %H:%M:%S'.
            style (str): style character '%' '{' or '$' for format string.

        """
        self.fmt = fmt
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)

    def format(self, record):
        if not hasattr(record, "fullName"):
            record.fullName = record.name
            record.name = record.name.split(".")[-1]

        return super().format(record)


class ColorFormatter(BaseFormatter):
    def __init__(
        self,
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt=None,
        style="%",
        level_colors=DEFAULT_LEVEL_COLORS,
        logger_names=None,
        logger_color=ANSIColor.CYAN,
    ):
        """ColorFormatter to format colors based on log levels. Optionally can color logger_names.

        Args:
            fmt (str): format string which uses LogRecord attributes.
            datefmt (str): date/time format string. Defaults to '%Y-%m-%d %H:%M:%S'.
            style (str): style character '%' '{' or '$' for format string.
            level_colors (Dict[str, str]): dict of levelname: ANSI color. Defaults to
                {
                    "DEBUG": ANSIColor.GREY,
                    "INFO": ANSIColor.GREY,
                    "WARNING": ANSIColor.YELLOW,
                    "ERROR": ANSIColor.RED,
                    "CRITICAL": ANSIColor.BOLD_RED,
                }
            logger_names (List[str]): list of logger names to apply logger_color.
            logger_color (int): ANSI custom color for logger_names.

        """
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.logger_names = logger_names or []
        self.logger_color = logger_color
        self.level_colors = level_colors

    def format(self, record):
        super().format(record)

        if record.levelno <= logging.INFO and any(
            record.name.startswith(logger_name) for logger_name in self.logger_names
        ):
            # Apply logger_color to logger_names
            log_fmt = ansi_sgr(self.logger_color) + self.fmt + ansi_sgr(ANSIColor.RESET)
        else:
            # Apply level_colors based on record levelname
            log_fmt = (
                ansi_sgr(self.level_colors.get(record.levelname, ANSIColor.GREY)) + self.fmt + ansi_sgr(ANSIColor.RESET)
            )

        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class JsonFormatter(BaseFormatter):
    def __init__(
        self, fmt="%(asctime)s - %(name)s - %(fullName)s - %(levelname)s - %(message)s", datefmt=None, style="%"
    ):
        """Format log records into JSON.

        Args:
            fmt (str): format string which uses LogRecord attributes. Attributes are used for JSON keys.
            datefmt (str): date/time format string. Defaults to '%Y-%m-%d %H:%M:%S'.
            style (str): style character '%' '{' or '$' for format string.

        """
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.fmt_dict = self.generate_fmt_dict(self.fmt)

    def generate_fmt_dict(self, fmt: str) -> dict:
        # Parse the `fmt` string and create a mapping of keys to LogRecord attributes
        matches = re.findall(r"%\((.*?)\)([sd])", fmt)

        fmt_dict = {}
        for key, _ in matches:
            if key == "shortname":
                fmt_dict["name"] = "shortname"
            else:
                fmt_dict[key] = key

        return fmt_dict

    def extract_bracket_fields(self, message: str) -> dict:
        # Extract bracketed fl_ctx_fields eg. [k1=v1, k2=v2...] into sub dict
        bracket_fields = {}
        match = re.search(r"\[(.*?)\]:", message)
        if match:
            pairs = match.group(1).split(", ")
            for pair in pairs:
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    bracket_fields[key] = value
        return bracket_fields

    def formatMessage(self, record) -> dict:
        return {fmt_key: record.__dict__.get(fmt_val, "") for fmt_key, fmt_val in self.fmt_dict.items()}

    def format(self, record) -> str:
        super().format(record)

        record.message = record.getMessage()
        bracket_fields = self.extract_bracket_fields(record.message)
        record.asctime = self.formatTime(record)

        formatted_message_dict = self.formatMessage(record)
        message_dict = {k: v for k, v in formatted_message_dict.items() if k != "message"}

        if bracket_fields:
            message_dict["fl_ctx_fields"] = bracket_fields
            record.message = re.sub(r"\[.*?\]:", "", record.message).strip()

        message_dict[self.fmt_dict.get("message", "message")] = record.message

        return json.dumps(message_dict, default=str)


class FLFilter(logging.Filter):
    def __init__(self, logger_names=["nvflare.app_common", "nvflare.app_opt"]):
        """Filter log records based on logger names.

        Args:
            logger_names (List[str]): list of logger names to allow through filter (inclusive)

        """
        super().__init__()
        self.logger_names = logger_names

    def filter(self, record):
        # Filter log records based on the logger name
        fullName = record.fullName if hasattr(record, "fullName") else record.name
        if any(fullName.startswith(name) for name in self.logger_names):
            return record.levelno >= logging.INFO


def ansi_sgr(code: str):
    """ANSI Select Graphics Rendition."""
    return "\x1b[" + code + "m"


def get_module_logger(module=None, name=None):
    if module is None:
        caller_globals = inspect.stack()[1].frame.f_globals
        module = caller_globals.get("__name__", "")

    return logging.getLogger(f"{module}.{name}" if name else module)


def get_obj_logger(obj):
    return logging.getLogger(f"{obj.__module__}.{obj.__class__.__qualname__}")


def get_script_logger():
    caller_frame = inspect.stack()[1]
    package = caller_frame.frame.f_globals.get("__package__", "")
    file = caller_frame.frame.f_globals.get("__file__", "")

    return logging.getLogger(
        f"{package + '.' if package else ''}{os.path.splitext(os.path.basename(file))[0] if file else ''}"
    )


def update_filenames(obj, dir_path: str = "", file_prefix: str = ""):
    """Update 'filename' keys in JSON objects with dir_path and file_prefix."""
    if "filename" in obj and isinstance(obj["filename"], str):
        filename = obj["filename"]
        if file_prefix:
            filename = os.path.join(os.path.dirname(filename), file_prefix + "_" + os.path.basename(filename))
        obj["filename"] = os.path.join(dir_path, filename)
    return obj


def read_log_config(file, dir_path: str = "", file_prefix: str = "") -> dict:
    """
    Reads JSON logging configuration file and returns config dictionary.
    Updates 'filename' keys with dir_path for dynamic locations.

    Args:
        file (str): Path to the configuration file.
        dir_path (str): Update filename keys with dir_path.

    Returns:
        config (dict)
    """
    try:
        with open(file, "r") as f:
            config = json.load(f, object_hook=lambda obj: update_filenames(obj, dir_path, file_prefix))
        return config
    except Exception as e:
        raise ValueError(f"Unrecognized logging configuration format. Failed to parse JSON: {e}.")


def configure_logging(workspace: Workspace, dir_path: str = "", file_prefix: str = ""):
    log_config_file_path = workspace.get_log_config_file_path()
    assert os.path.isfile(log_config_file_path), f"missing log config file {log_config_file_path}"

    dict_config = read_log_config(log_config_file_path, dir_path, file_prefix)
    logging.config.dictConfig(dict_config)


def add_log_file_handler(log_file_name):
    root_logger = logging.getLogger()
    main_handler = root_logger.handlers[0]
    file_handler = RotatingFileHandler(log_file_name, maxBytes=20 * 1024 * 1024, backupCount=10)
    file_handler.setLevel(main_handler.level)
    file_handler.setFormatter(main_handler.formatter)
    root_logger.addHandler(file_handler)


def print_logger_hierarchy(package_name="nvflare", level_colors=DEFAULT_LEVEL_COLORS):
    all_loggers = logging.root.manager.loggerDict

    # Filter for package loggers based on package_name
    package_loggers = {name: logger for name, logger in all_loggers.items() if name.startswith(package_name)}
    sorted_package_loggers = sorted(package_loggers.keys())

    # Print package loggers with hierarcjy
    print(f"hierarchical loggers ({len(package_loggers)}):")

    def get_effective_level(logger_name):
        # Search for effective level from parent loggers
        parts = logger_name.split(".")
        for i in range(len(parts), 0, -1):
            parent_name = ".".join(parts[:i])
            parent_logger = package_loggers.get(parent_name)
            if isinstance(parent_logger, Logger) and parent_logger.level != logging.NOTSET:
                return logging.getLevelName(parent_logger.level)

        # If no parent has a set level, default to the root logger's effective level
        return logging.getLevelName(logging.root.level)

    def print_hierarchy(logger_name, indent_level=0):
        logger = package_loggers.get(logger_name)
        level_name = get_effective_level(logger_name)

        # Indicate "(unset)" placeholders if logger.level == NOTSET
        is_unset = isinstance(logger, Logger) and logger.level == logging.NOTSET or not isinstance(logger, Logger)
        level_display = f"{level_name} (SET)" if not is_unset else level_name

        # Print the logger with color and indentation
        color = level_colors.get(level_name, ANSIColor.RESET)
        print("    " * indent_level + f"{ansi_sgr(color)}{logger_name} [{level_display}]{ansi_sgr(ANSIColor.RESET)}")

        # Find child loggers based on the current hierarchy level
        for name in sorted_package_loggers:
            if name.startswith(logger_name + ".") and name.count(".") == logger_name.count(".") + 1:
                print_hierarchy(name, indent_level + 1)

    print_hierarchy(package_name)
