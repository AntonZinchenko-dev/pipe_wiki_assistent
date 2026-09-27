"""Тесты идут на настройках из кода, а не из локального `.env`.

Без этого `.env` разработчика (GigaChat, агент, другой промпт) молча
подмешивается в тесты, и они проверяют чужую конфигурацию.
"""

import os

from app.config import Settings

Settings.model_config["env_file"] = None
for _name in [name for name in os.environ if name.upper().startswith("PW_")]:
    del os.environ[_name]
