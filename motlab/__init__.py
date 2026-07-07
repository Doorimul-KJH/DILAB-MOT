"""Local bootstrap package for running ``python -m motlab...`` from source.

The real package code lives under ``src/motlab``. This small package extends
the import path so command-line usage works from a fresh checkout without an
editable install.
"""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "motlab"
if SRC_PACKAGE.exists() and str(SRC_PACKAGE) not in __path__:
    __path__.append(str(SRC_PACKAGE))
