"""Central logging setup for hepml.

Every module obtains its logger via get_logger(__name__); the first call
configures the root handler. Format is ASCII-only so output survives
Windows cp1252 consoles.
"""

from __future__ import annotations

import logging

_CONFIGURED = False


def setup(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(level=level, format="[%(levelname)s] %(name)s: %(message)s")
        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup()
    return logging.getLogger(name)
