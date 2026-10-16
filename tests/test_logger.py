"""The project logger (step 9.5): console shows the plain message, the file gets time, level and module."""

import logging

from src.utils import logger as logger_module
from src.utils.logger import get_logger


def test_logger_writes_plain_console_and_detailed_file(tmp_path, monkeypatch, capsys):
    root = logging.getLogger(logger_module.ROOT_NAME)
    saved = root.handlers[:]
    root.handlers.clear()
    monkeypatch.setattr("src.utils.helpers.resolve_path", lambda p: tmp_path / p)
    try:
        log = get_logger("src.training.train_example")
        assert log.name == "fincast.training.train_example"
        log.info("Saved models/example.json")
        log.warning("Careful: %s", "something")
        for handler in root.handlers:
            handler.flush()
        out = capsys.readouterr().out
        assert "Saved models/example.json\n" in out and "Careful: something" in out
        text = (tmp_path / "logs" / "fincast.log").read_text()
        assert "INFO    fincast.training.train_example: Saved models/example.json" in text
        assert "WARNING fincast.training.train_example: Careful: something" in text
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers[:] = saved


def test_no_print_left_in_pipeline_code():
    from src.utils.helpers import get_project_root
    root = get_project_root()
    for folder in ("src", "pipelines"):
        for path in (root / folder).rglob("*.py"):
            assert "print(" not in path.read_text().replace("def _print(", "").replace("_print(", ""), path
