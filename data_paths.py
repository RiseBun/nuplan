"""Project-local data path defaults.

All data locations can be overridden with environment variables so the project
does not depend on one machine's mounted `/mnt/...` layout.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def _split_paths(value: str) -> List[str]:
    # Support both POSIX pathsep and comma-separated values for shell convenience.
    parts: List[str] = []
    for chunk in value.split(os.pathsep):
        parts.extend(item.strip() for item in chunk.split(","))
    return [item for item in parts if item]


DATA_ROOT = _path_from_env(
    "NUPLAN_DATA_ROOT",
    WORKSPACE_ROOT,
)
DB_ROOT = _path_from_env(
    "NUPLAN_DB_ROOT",
    WORKSPACE_ROOT / "data" / "cache" / "mini",
)
INDEX_ROOT = _path_from_env("NUPLAN_INDEX_ROOT", PROJECT_ROOT / "indices_v4")
GENERATED_DATA_ROOT = _path_from_env(
    "NUPLAN_GENERATED_DATA_ROOT",
    PROJECT_ROOT / "generated_data",
)
LABELED_DATA_ROOT = _path_from_env(
    "NUPLAN_LABELED_DATA_ROOT",
    PROJECT_ROOT / "labeled_data",
)
def camera_roots(data_root: Path = DATA_ROOT) -> List[Path]:
    value = os.environ.get("NUPLAN_CAMERA_ROOTS")
    if value:
        return [Path(item).expanduser() for item in _split_paths(value)]

    cam0 = os.environ.get("NUPLAN_CAMERA_0_ROOT")
    cam1 = os.environ.get("NUPLAN_CAMERA_1_ROOT")
    if cam0 or cam1:
        roots = []
        if cam0:
            roots.append(Path(cam0).expanduser())
        if cam1:
            roots.append(Path(cam1).expanduser())
        return roots

    discovered = sorted(data_root.glob("nuplan-v1.1_mini_camera_*"))
    if discovered:
        return [path for path in discovered if path.is_dir()]

    return [
        data_root / "nuplan-v1.1_mini_camera_0",
        data_root / "nuplan-v1.1_mini_camera_1",
    ]


def path_str(path: Path) -> str:
    return str(path)
