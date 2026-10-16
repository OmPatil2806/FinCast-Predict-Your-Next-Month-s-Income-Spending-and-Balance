"""One logger setup for the whole project (step 9.5).

    log = get_logger(__name__)
    log.info("Saved models/champions.json")

Every logger is a child of "fincast". The first call attaches two handlers to it:
    console  the message only, to stdout - pipeline output looks exactly like before
    file     time, level, module and message in logs/fincast.log (paths.logs in config.yaml),
             rotated at 1 MB with 3 old files kept, so it never grows without limit
If the log folder cannot be written (a read-only deployment), logging falls back to the console only.
Set FINCAST_LOG_LEVEL=DEBUG for more detail.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

ROOT_NAME = "fincast"
LOG_FILE = "fincast.log"
MAX_BYTES = 1_000_000
BACKUPS = 3
FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def _configure(root: logging.Logger) -> None:
    from src.utils.helpers import load_config, resolve_path
    root.setLevel(os.environ.get("FINCAST_LOG_LEVEL", "INFO").upper())
    root.propagate = False
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)
    try:
        folder = resolve_path(load_config()["paths"].get("logs", "logs"))
        folder.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(folder / LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8")
        handler.setFormatter(logging.Formatter(FILE_FORMAT))
        root.addHandler(handler)
    except OSError as err:
        root.warning("Logging to the console only (%s)", err)


def get_logger(name: str) -> logging.Logger:
    """A project logger: 'src.training.train_baselines' -> 'fincast.training.train_baselines'."""
    root = logging.getLogger(ROOT_NAME)
    if not root.handlers:
        _configure(root)
    if name == "__main__":                       # a script started directly: use its file name
        main_file = getattr(sys.modules.get("__main__"), "__file__", None) or "main"
        name = os.path.splitext(os.path.basename(main_file))[0]
    short = name.removeprefix("src.").removeprefix("pipelines.")
    return root.getChild(short)
