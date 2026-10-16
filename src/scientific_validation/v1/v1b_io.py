"""Minimal artifact IO for the flattened V1-B package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple
import json

import numpy as np
import torch



LEVEL_DIRS = {
    "Z": "dbrl",
    "S": "bse",
    "S_tilde": "bil",
    "g": "ebrl",
}

REPRESENTATION_KEY = "representations"


@dataclass
class ShardRecord:
    split: str
    shard_index: int
    start_index: int
    end_index: int
    paths: Dict[str, Path]


def _extract_tensor(obj, REPRESENTATION_KEY):
    if not isinstance(obj, dict):
        raise TypeError(
            f"Expected shard object to be a dict, "
            f"received {type(obj).__name__}"
        )

    if REPRESENTATION_KEY in obj:
        tensor = obj[REPRESENTATION_KEY]

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"'{REPRESENTATION_KEY}' must be a torch.Tensor, "
                f"received {type(tensor).__name__}"
            )

        return tensor

    raise KeyError(
        f"Could not find '{REPRESENTATION_KEY}' in representation shard. "
        f"Available keys={tuple(obj.keys())}"
    )


def load_representation_shard(
    artifact_root: Path,
    split: str,
    shard_index: int,
    *,
    return_metadata: bool = False,
    levels: Optional[Iterable[str]] = None,
    expected_checkpoint_sha256: str | None = None,
):
    """Load one HDEG representation shard.

    The persisted HDEG schema stores the tensor under ``representations``.
    When ``return_metadata`` is True, the artifact metadata is returned for
    each level so the V1 eligibility layer can verify sample ranges and
    provenance without inventing a parallel schema.
    """
    result = {}
    metadata = {}
    requested_levels = tuple(LEVEL_DIRS) if levels is None else tuple(levels)
    unknown = [level for level in requested_levels if level not in LEVEL_DIRS]
    if unknown:
        raise ValueError(f"Unknown representation level(s): {unknown}")

    for level in requested_levels:
        dirname = LEVEL_DIRS[level]
        p = Path(f"{artifact_root}/{dirname}/{split}/shard_{shard_index:06d}.pt")
        if not p.exists():
            raise FileNotFoundError(p)
        obj = torch.load(p, map_location="cpu", weights_only=False)
        if expected_checkpoint_sha256 is not None:
            stages = tuple(LEVEL_DIRS.values())
            for field in ("model_checkpoint_sha256",) + tuple(
                f"source_{stage}_checkpoint_sha256"
                for stage in stages[:stages.index(dirname)]
            ):
                if obj.get(field) != expected_checkpoint_sha256:
                    raise RuntimeError(f"{p}: checkpoint mismatch in {field}")
            for stage in stages[:stages.index(dirname)]:
                if obj.get(f"source_{stage}_shard_index") != shard_index:
                    raise RuntimeError(f"{p}: source_{stage}_shard_index mismatch")
        tensor = _extract_tensor(obj, REPRESENTATION_KEY)
        result[level] = tensor.detach().cpu().numpy()
        if return_metadata:
            metadata[level] = {k: v for k, v in obj.items() if k != REPRESENTATION_KEY}

    if return_metadata:
        return result, metadata
    return result


def discover_shards(
    artifact_root: Path,
    split: str,
) -> list[int]:
    all_ids = []
    for stage in LEVEL_DIRS.values():
        directory = Path(artifact_root) / stage / split
        ids = []
        for p in directory.glob("shard_*.pt"):
            text = p.stem.removeprefix("shard_")
            if not text.isdigit():
                raise ValueError(f"Invalid shard name: {p}")
            ids.append(int(text))
        ids.sort()
        if not ids or ids != list(range(len(ids))):
            raise RuntimeError(f"Missing/non-contiguous shards: {directory}")
        all_ids.append(ids)
    if any(ids != all_ids[0] for ids in all_ids[1:]):
        raise RuntimeError(f"Cross-level shard counts differ for {split}")
    return all_ids[0]


def load_window_shard(
    window_root: Path,
    split: str,
    shard_index: int,
) -> dict:
    p = Path(f"{window_root}/{split}/shard_{shard_index:06d}.npz")
    if not p.exists():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    return out


def save_json(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)
