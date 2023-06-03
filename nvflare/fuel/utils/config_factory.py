import os
import pathlib
from typing import Optional, List

from nvflare.fuel.utils.config import ConfigFormat, Config
from nvflare.fuel.utils.config_loader import ConfigLoader
from nvflare.fuel.utils.json_config_loader import JsonConfigLoader
from nvflare.fuel.utils.omegaconf_loader import OmegaConfLoader
from nvflare.fuel.utils.pyhocon_loader import PyhoconLoader


class ConfigFactory:
    config_format_search_order = [ConfigFormat.JSON,ConfigFormat.JSON_DEFAULT,
                                  ConfigFormat.PYHOCON,ConfigFormat.PYHOCON_DEFAULT,
                                  ConfigFormat.OMEGACONF, ConfigFormat.OMEGACONF_DEFAULT]

    @staticmethod
    def config_exts() -> str:
        exts = [f.value for f in ConfigFactory.config_format_search_order]
        return "|".join(exts)

    @staticmethod
    def search_config_format(init_file_path,
                             search_dirs: Optional[List[str]] = None) -> (Optional[ConfigFormat], Optional[str]):
        # we ignore the original extension
        if not search_dirs:
            parent_dir = pathlib.Path(init_file_path).parent
            search_dirs = [parent_dir]
        file_path = os.path.splitext(pathlib.Path(init_file_path).name)[0]
        for fmt in ConfigFactory.config_format_search_order:
            print(f"search format {fmt.name} with ext {fmt.value},file:{file_path}, search dirs = {search_dirs}")
            for search_dir in search_dirs:
                for root, dirs, files in os.walk(search_dir):
                    file = f"{file_path}{fmt.value}"
                    print("file=", file)
                    print("files=", files)
                    if file in files:
                        config_file_path = os.path.join(root, file)
                        return fmt, config_file_path
        print(f"None for {file_path}, dirs = {search_dirs}")
        return None, None

    @staticmethod
    def has_config(init_file_path: str, search_dirs: Optional[List[str]] = None) -> bool:
        _, real_file_path = ConfigFactory.search_config_format(init_file_path, search_dirs)
        return real_file_path is not None

    @staticmethod
    def match_config(parent, init_file_path, match_fn) -> bool:
        # we ignore the original extension
        basename = os.path.splitext(init_file_path)[0]
        for fmt in ConfigFactory.config_format_search_order:
            print(f"search format {fmt.name} with ext {fmt.value} in {basename}{fmt.value}")
            if match_fn(parent, f"{basename}{fmt.value}"):
                return True
        return False

    @staticmethod
    def load_config(file_path: str, search_dirs: Optional[List[str]] = None) -> Config:
        config_format, real_config_file_path = ConfigFactory.search_config_format(file_path, search_dirs)
        print(config_format, real_config_file_path)
        config_loader = ConfigFactory.get_config_loader(config_format)
        conf = config_loader.load_config(real_config_file_path)
        print("conf is none = ", conf is None)
        return conf

    @staticmethod
    def get_config_loader(config_format: ConfigFormat) -> ConfigLoader:
        if config_format == ConfigFormat.JSON or config_format == ConfigFormat.JSON_DEFAULT:
            return JsonConfigLoader()
        elif config_format == ConfigFormat.OMEGACONF or config_format == ConfigFormat.OMEGACONF_DEFAULT:
            return OmegaConfLoader()
        elif config_format == ConfigFormat.PYHOCON or  config_format == ConfigFormat.PYHOCON_DEFAULT :
            return PyhoconLoader()
        else:
            raise NotImplemented(
                f"configuration format {config_format.name} with file ext {config_format.value} is not implemented")
