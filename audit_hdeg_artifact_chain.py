from __future__ import annotations

"""
HDEG V1.0 artifact-chain audit.

This tool audits the persisted scientific artifacts without loading a complete
split into RAM. It checks:

- shard discovery and contiguous ranges;
- representation shapes/dtypes/finiteness;
- DBRL -> BSE -> BIL -> EBRL provenance;
- HBF upstream alignment and provenance;
- MBAI t -> t+1 alignment;
- MBAI coverage and final-row exclusion;
- consistency of N/K/D/window size;
- presence of the HDEG training checkpoint;
- HBF checkpoint provenance.

It produces a machine-readable audit JSON and a human-readable report.

Scope:
PASS means that persisted artifact structure, recorded checkpoint provenance,
and temporal coverage are consistent. It does not establish scientific
representation quality or independently reproduce model inference.
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
import yaml

from src.utils.seed import set_seed
from src.utils.get_folders_utils import get_processed_folder

LEVELS = ("Z", "S", "S_tilde", "g")
STAGES = ("dbrl", "bse", "bil", "ebrl", "hbf", "mbai")
SPLITS = ("train", "val", "actor2_test", "actor1_test")


def shard_index(path: Path) -> int:
    m = re.fullmatch(r"shard_(\d+)\.pt", path.name)
    if not m:
        raise ValueError(f"Invalid shard filename: {path}")
    return int(m.group(1))


def discover(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("shard_*.pt"), key=shard_index)
    if not paths:
        raise FileNotFoundError(f"No shards found: {directory}")
    ids = [shard_index(p) for p in paths]
    if ids != list(range(len(paths))):
        raise RuntimeError(f"Non-contiguous shard indices in {directory}: {ids[:20]}")
    return paths


def load(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not contain a dictionary payload")
    return payload


def sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class Audit:
    def __init__(self):
        self.checks = []
        self.errors = []
        self.warnings = []

    def ok(self, name, detail=""):
        self.checks.append({"name": name, "status": "PASS", "detail": detail})

    def fail(self, name, detail):
        self.checks.append({"name": name, "status": "FAIL", "detail": detail})
        self.errors.append({"name": name, "detail": detail})

    def warn(self, name, detail):
        self.checks.append({"name": name, "status": "WARN", "detail": detail})
        self.warnings.append({"name": name, "detail": detail})


def validate_rep_payload(
    payload,
    path,
    *,
    split,
    level,
    N,
    K,
    D,
    expected_start,
    require_num_states,
):
    required = [
        "representations", "split", "shard_index",
        "start_index", "end_index", "window_size",
        "num_devices", "embedding_dim",
    ]
    if require_num_states:
        required.append("num_states")

    missing = [k for k in required if k not in payload]
    if missing:
        raise RuntimeError(f"{path}: missing metadata {missing}")

    if payload["split"] != split:
        raise RuntimeError(f"{path}: split mismatch")

    i = int(payload["shard_index"])
    if i != shard_index(path):
        raise RuntimeError(f"{path}: shard index metadata mismatch")

    start, end = int(payload["start_index"]), int(payload["end_index"])
    if start != expected_start or end <= start:
        raise RuntimeError(
            f"{path}: invalid/non-contiguous range [{start},{end}), "
            f"expected start {expected_start}"
        )

    x = payload["representations"]
    if not isinstance(x, torch.Tensor):
        raise RuntimeError(f"{path}: representations is not a tensor")
    if x.dtype != torch.float32:
        raise RuntimeError(f"{path}: representations dtype is {x.dtype}, expected float32")
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{path}: representations contains NaN/Inf")

    expected_ndim = 2 if level == "g" else 3
    if x.ndim != expected_ndim:
        raise RuntimeError(f"{path}: {level} ndim={x.ndim}, expected {expected_ndim}")

    if x.shape[0] != end - start:
        raise RuntimeError(f"{path}: sample count/range mismatch")

    if level == "Z":
        if tuple(x.shape[1:]) != (N, D):
            raise RuntimeError(f"{path}: Z shape {tuple(x.shape)}, expected (*,{N},{D})")
    elif level in ("S", "S_tilde"):
        if tuple(x.shape[1:]) != (K, D):
            raise RuntimeError(f"{path}: {level} shape {tuple(x.shape)}, expected (*,{K},{D})")
    else:
        if tuple(x.shape[1:]) != (D,):
            raise RuntimeError(f"{path}: g shape {tuple(x.shape)}, expected (*,{D})")

    if int(payload["num_devices"]) != N:
        raise RuntimeError(f"{path}: N metadata mismatch")
    if require_num_states and int(payload["num_states"]) != K:
        raise RuntimeError(f"{path}: K metadata mismatch")
    if int(payload["embedding_dim"]) != D:
        raise RuntimeError(f"{path}: D metadata mismatch")

    return start, end, int(payload["window_size"])


def validate_provenance(payload, path, stage, checkpoint_sha256, index):
    upstream = STAGES[:STAGES.index(stage)]
    fields = ["model_checkpoint_sha256"] + [
        f"source_{source}_checkpoint_sha256" for source in upstream
    ]
    for field in fields:
        if payload.get(field) != checkpoint_sha256:
            raise RuntimeError(
                f"{path}: {field} mismatch: expected {checkpoint_sha256}, "
                f"got {payload.get(field)!r}"
            )
    for source in upstream:
        field = f"source_{source}_shard_index"
        if int(payload.get(field, -1)) != index:
            raise RuntimeError(f"{path}: {field} mismatch")


def audit_split(base: Path, split: str, audit: Audit, checkpoint_sha256: str):
    # The source manifest prevents an equally truncated set of all six stages
    # from passing as a complete split. Window arrays need not be loaded.
    manifest_path = base / "windows" / split / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    M = int(manifest["num_samples"])
    shard_size = int(manifest["shard_size"])
    shard_count = int(manifest["num_shards"])
    N, W = int(manifest["num_devices"]), int(manifest["window_size"])
    if manifest["split"] != split or min(M, shard_size, shard_count, N, W) <= 0:
        raise RuntimeError(f"Invalid source window manifest: {manifest_path}")
    if shard_count != (M + shard_size - 1) // shard_size:
        raise RuntimeError("Window manifest sample/shard counts disagree")
    names = [Path(p).name for p in manifest["shards"]]
    expected_names = [f"shard_{i:06d}.npz" for i in range(shard_count)]
    if names != expected_names:
        raise RuntimeError("Window manifest shard list is inconsistent")
    actual_names = sorted(p.name for p in manifest_path.parent.glob("shard_*.npz"))
    if actual_names != names:
        raise RuntimeError("Window files differ from the source manifest shard list")
    audit.ok("window_manifest", f"{M} windows, {shard_count} shards; W={W}, N={N}")

    paths = {stage: discover(base / stage / split) for stage in STAGES[:-1]}
    # A final representation shard containing one row has no MBAI target.
    mbai_count = shard_count - int(M - (shard_count - 1) * shard_size == 1)
    mbai_dir = base / "mbai" / split
    paths["mbai"] = (
        discover(mbai_dir) if mbai_count else sorted(mbai_dir.glob("shard_*.pt"))
    )
    counts = {stage: len(items) for stage, items in paths.items()}
    for stage, count in counts.items():
        required = mbai_count if stage == "mbai" else shard_count
        if count != required:
            raise RuntimeError(f"{stage}: expected {required} shards; found {count}")
    audit.ok("shard_count_consistency", str(counts))

    first = load(paths["dbrl"][0])
    D = int(first["embedding_dim"])
    del first
    first = load(paths["bse"][0])
    K = int(first["num_states"])
    del first
    if min(D, K) <= 0:
        raise RuntimeError("Embedding/state dimensions must be positive")

    total_m = 0
    weights_reference = None
    for i in range(shard_count):
        start, end = i * shard_size, min((i + 1) * shard_size, M)
        for level, stage in zip(LEVELS, STAGES[:4]):
            path = paths[stage][i]
            payload = load(path)
            validate_provenance(payload, path, stage, checkpoint_sha256, i)
            got_start, got_end, got_W = validate_rep_payload(
                payload, path, split=split, level=level, N=N, K=K, D=D,
                expected_start=start, require_num_states=(stage != "dbrl"),
            )
            if (got_start, got_end, got_W) != (start, end, W):
                raise RuntimeError(f"{path}: range/window differs from source manifest")
            del payload

        path = paths["hbf"][i]
        h = load(path)
        validate_provenance(h, path, "hbf", checkpoint_sha256, i)
        for level in LEVELS:
            # Reuse structural validation without copying tensor storage.
            proxy = {**h, "representations": h[level]}
            hs, he, hw = validate_rep_payload(
                proxy, path, split=split, level=level, N=N, K=K, D=D,
                expected_start=start, require_num_states=True,
            )
            if (hs, he, hw) != (start, end, W):
                raise RuntimeError(f"{path}: HBF range/window mismatch")
            del proxy
        del h

        if i < mbai_count:
            path = paths["mbai"][i]
            m = load(path)
            validate_provenance(m, path, "mbai", checkpoint_sha256, i)
            ts, te = start + 1, min(end + 1, M)
            expected_meta = {
                "split": split, "shard_index": i, "window_size": W,
                "num_devices": N, "num_states": K, "embedding_dim": D,
                "source_hbf_start_index": start, "source_hbf_end_index": end,
                "target_start_index": ts, "target_end_index": te,
                "start_index": ts, "end_index": te, "num_samples": te - ts,
                "final_prediction_excluded": i == shard_count - 1,
                "last_prediction_excluded": i == shard_count - 1,
                "run_truncated": False,
            }
            for field, expected in expected_meta.items():
                if m.get(field) != expected:
                    raise RuntimeError(
                        f"{path}: {field}: expected {expected!r}, got {m.get(field)!r}"
                    )
            for key in ("E_Z", "E_S", "E_S_tilde", "E_G", "A"):
                x = m[key]
                if not isinstance(x, torch.Tensor) or x.dtype != torch.float32:
                    raise RuntimeError(f"{path}: {key} must be a float32 tensor")
                if tuple(x.shape) != (te - ts,) or not torch.isfinite(x).all():
                    raise RuntimeError(f"{path}: {key} shape/finiteness mismatch")
                if (x < 0).any():
                    raise RuntimeError(f"{path}: {key} contains negative anomaly evidence")
            weights = m["fusion_weights"]
            if not isinstance(weights, dict) or set(weights) != set(LEVELS):
                raise RuntimeError(f"{path}: invalid fusion weight keys")
            if any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or v < 0 for v in weights.values()):
                raise RuntimeError(f"{path}: invalid fusion weights")
            if sum(weights.values()) <= 0:
                raise RuntimeError(f"{path}: fusion weights sum must be positive")
            if weights_reference is not None and weights != weights_reference:
                raise RuntimeError(f"{path}: fusion weights changed between shards")
            weights_reference = weights
            total_m += te - ts
            del m, x
        print(f"[PASS] artifact shard {i + 1}/{shard_count}: ranges, tensors, provenance")

    if total_m != M - 1:
        raise RuntimeError(f"Expected {M-1} aligned assessments; got {total_m}")
    audit.ok("representation_chain", f"{M} samples; N={N}, K={K}, D={D}, W={W}")
    audit.ok("hbf_artifacts", f"{M} prediction rows aligned with source shards")
    audit.ok("checkpoint_provenance", f"All stages and recorded ancestors match {checkpoint_sha256}")
    audit.ok("mbai_temporal_coverage", f"{total_m} aligned assessments; target range [1,{M})")
    audit.ok("final_prediction_boundary", "exactly one final unpaired prediction excluded")
    return M, total_m


def main():

    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root_dir", type=str)
    parser.add_argument("--split", choices=SPLITS)
    args = parser.parse_args()
    if args.project_root_dir is not None:
        config["project_root_dir"] = args.project_root_dir
    if args.split is not None:
        config["split"] = args.split
    split = config.get("split")
    if split not in SPLITS:
        raise ValueError(f"Set config split or --split to one of {SPLITS}.")
    set_seed(int(config["seed"]))
    base = Path(get_processed_folder(config))
    dataset = base.name
    output = base / "audit_reports" / f"artifact_audit_{split}.json"
    a = Audit()
    total = total_m = None
    checkpoint_path = None
    checkpoint_sha256 = None
    try:
        checkpoint = config["hdeg"]["hbf"].get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError("hdeg.hbf.checkpoint must name an HDEG checkpoint.")
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = base / "hdeg_checkpoints" / checkpoint_path
        checkpoint_sha256 = sha256(checkpoint_path)
        a.ok("selected_checkpoint", str(checkpoint_path))
        print(f"Split: {split}")
        print(f"Checkpoint SHA256: {checkpoint_sha256}")
        total, total_m = audit_split(base, split, a, checkpoint_sha256)
    except Exception as exc:
        a.fail("artifact_chain", str(exc))

    report = {
        "audit_scope": "artifact structure, recorded provenance, and full-split temporal coverage",
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset": dataset,
        "split": split,
        "total_prediction_rows": total,
        "total_aligned_assessments": total_m,
        "status": "PASS" if not a.errors else "FAIL",
        "checks": a.checks,
        "errors": a.errors,
        "warnings": a.warnings,
    }

    out = output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Audit report: {out}")
    raise SystemExit(0 if not a.errors else 2)


if __name__ == "__main__":
    main()
