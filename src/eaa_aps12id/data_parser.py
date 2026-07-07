from __future__ import annotations

from os import PathLike
from pathlib import Path

import numpy as np


class SAXSDataParser:
    """Parse averaged SAXS ``.dat`` files into numeric q/intensity/error data."""

    def parse(self, file_path: str | PathLike[str]) -> np.ndarray:
        """Return SAXS data as an ``(n, 3)`` array of q, intensity, and error."""
        path = Path(file_path)
        if path.suffix != ".dat":
            raise ValueError(f"Expected a .dat file, got: {path}")

        data = np.loadtxt(path, comments="%")
        data = np.atleast_2d(data)

        if data.shape[1] != 3:
            raise ValueError(
                f"Expected three data columns (q, intensity, error), got {data.shape[1]}"
            )

        return data
