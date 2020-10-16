import os
import toml


class Config:
    def __init__(self, config_file=None):
        self.elements = self._load(config_file)

    def _load(self, path):
        return toml.load(path)
