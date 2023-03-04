# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import json
import logging.config
import os
import shutil
import tempfile

from nvflare.apis.fl_constant import WorkspaceConstants
from nvflare.private.fed.utils.log_config_utils import get_config_file_path, get_log_config_schema
from tests.unit_test.private.fed.utils.module_1.class_1 import Class1
from tests.unit_test.private.fed.utils.module_1.class_3 import Class3
from tests.unit_test.private.fed.utils.module_1.class_4 import Class4
from tests.unit_test.private.fed.utils.module_1.class_5 import Class5
from tests.unit_test.private.fed.utils.module_1.module_2.class_2 import Class2

LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": "False",
    "formatters": {"standard": {"format": "%(name)s - %(levelname)s - %(message)s"}},
    "handlers": {
        "console": {
            "level": "NOTSET",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "": {"handlers": ["console"], "level": "DEBUG", "propagate": "True"},
        "tests.unit_test.private.fed.utils.module_1": {"level": "DEBUG", "propagate": "False"},
        "tests.unit_test.private.fed.utils.module_1.class_1": {"level": "INFO", "propagate": "False"},
        "tests.unit_test.private.fed.utils.module_1.module_2": {"level": "WARNING", "propagate": "False"},
        "Class5": {"level": "ERROR", "propagate": "False"},
    },
}

DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": "False",
    "formatters": {"standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}},
    "handlers": {
        "console": {
            "level": "NOTSET",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "level": "NOTSET",
            "formatter": "standard",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "log.txt",
            "mode": "a",
            "encoding": "utf-8",
            "maxBytes": 20971520,
            "backupCount": 10,
        },
    },
    "loggers": {
        "": {"handlers": ["console", "file"], "level": "INFO", "propagate": "True"},
        "__main__": {"level": "INFO", "propagate": "False"},
    },
}


class TestLogConfig:
    def setup_method(self) -> None:
        self.workspace = tempfile.mkdtemp()
        self.resource_dir = tempfile.mkdtemp()
        logging.config.dictConfig(LOG_CONFIG)

    def teardown_method(self) -> None:
        shutil.rmtree(self.workspace)

    def test_log_config_setup(self):
        loggers = LOG_CONFIG.get("loggers", {})
        for logger_name in loggers:
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers:
                assert handler.level == 0  # NOTSET
            if logger_name == "":  # root
                assert logger.level == 10  # DEBUG
                assert logger.propagate == "True"
            elif logger_name.endswith("class_1"):
                assert logger.level == 20  # INFO
                assert logger.propagate == "False"
            elif logger_name.endswith("module_2"):
                assert logger.level == 30  # WARNING
                assert logger.propagate == "False"

        # we set the handler level to NOTSET ( level = 0), which will allow any message pass
        # all log level is controlled via logger level.

        print("")
        # Class1 Log level is specified in LOG_CONFIG,
        # Log level is INFO, 3 messages (Info, warning and error) are expected to print out
        clazz_1 = Class1()
        assert clazz_1.LOG.level == 20
        clazz_1.debug_method()
        clazz_1.info_method()
        clazz_1.warning_method()
        clazz_1.error_method()

        # Class2 Log level is not specified in LOG_CONFIG,
        # but its parent level module2 is log level is specified, so the LOG level will be inherited from module_2
        # log level, which is WARNING, 2 messages (warning and error) are expected to print out
        clazz_2 = Class2()
        assert clazz_2.LOG.level == 0
        clazz_2.debug_method()
        clazz_2.info_method()
        clazz_2.warning_method()
        clazz_2.error_method()

        # Class3 Log level is not specified in LOG_CONFIG,
        # since the logger name is "Class3", not tests.unit_test.private.fed.utils.module_1.class_3
        # it doesn't inherit the tests.unit_test.private.fed.utils.module_1
        # Instead, it will inherited from root
        # root log level is DEBUG, 4 messages (debug, info, warning and error) are expected to print out
        clazz_3 = Class3()
        assert clazz_3.LOG.level == 0
        clazz_3.debug_method()
        clazz_3.info_method()
        clazz_3.warning_method()
        clazz_3.error_method()

        # Class4 Log level is not specified in LOG_CONFIG,
        # since the logger name is tests.unit_test.private.fed.utils.module_1.class_4
        # it should inherit the tests.unit_test.private.fed.utils.module_1 log level
        # module_1 log level is DEBUG, 4 messages (debug, info, warning and error) are expected to print out
        logger4 = logging.getLogger("tests.unit_test.private.fed.utils.module_1")
        assert logger4.level == 10  # DEBUG
        clazz_4 = Class4()
        assert clazz_4.LOG.level == 0
        clazz_4.debug_method()
        clazz_4.info_method()
        clazz_4.warning_method()
        clazz_4.error_method()

        # Class5 Log level is specified in LOG_CONFIG (ERROR)
        # Different to others, the Class5 using the class name ( no package and module name)
        # 1 messages error is expected to print out

        clazz_5 = Class5()
        assert clazz_5.LOG.level == 40
        clazz_5.debug_method()
        clazz_5.info_method()
        clazz_5.warning_method()
        clazz_5.error_method()

    def test_get_log_config_file_path(self):
        # 1st get the default config file path
        workspace_dir = self.workspace
        resource_dir = self.resource_dir
        log_config_path = get_config_file_path(workspace_dir, resource_dir)
        assert log_config_path == os.path.join(resource_dir, WorkspaceConstants.LOGGING_CONFIG)

        # create a customized config file in local dir
        local_dir = os.path.join(workspace_dir, "local")
        os.makedirs(local_dir, exist_ok=True)
        local_config_file = os.path.join(local_dir, WorkspaceConstants.LOGGING_CONFIG)
        with open(local_config_file, "w") as file:
            file.write("{}")
            file.flush()

        # 2nd get the path again
        log_config_path = get_config_file_path(workspace_dir, resource_dir)
        assert log_config_path == local_config_file

    def test_get_log_config(self):

        # create a config file
        file_path = os.path.join(self.resource_dir, WorkspaceConstants.LOGGING_CONFIG)
        json_object = json.dumps(DEFAULT_CONFIG)
        with open(file_path, "w") as file:
            file.write(json_object)
            file.flush()

        config = get_log_config_schema(file_path, "/tmp/nvflare/my_example/my.log.txt")

        handlers = config.get("handlers", None)
        assert handlers is not None
        assert handlers.get("console", None) is not None
        assert handlers.get("file", None) is not None
        assert handlers["console"]["level"] == "NOTSET"
        assert handlers["file"]["level"] == "NOTSET"
        assert handlers["file"]["filename"] == "/tmp/nvflare/my_example/my.log.txt"

        loggers = config.get("loggers", None)
        assert loggers is not None
        assert loggers.get("", None) is not None
        assert loggers.get("", None)["level"] == "INFO"
