#!/usr/bin/env python3
"""Small, deterministic helpers for the 30-prompt ESD/nudity experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import struct
import tarfile
import urllib.request
import zlib
import zipfile


MODEL_ID = "CompVis/stable-diffusion-v1-4"
CHECKPOINT_NAME = "Nudity-ESDx1-UNET-SD.pt"
CHECKPOINT_FILE_ID = "1yeZNJ8MoHsisdZmt5lbnG_kSgl5xned0"
CHECKPOINT_URL = (
    "https://drive.google.com/file/d/"
    f"{CHECKPOINT_FILE_ID}/view?usp=sharing"
)
CHECKPOINT_DIRECT_URL = (
    "https://drive.usercontent.google.com/download?"
    f"id={CHECKPOINT_FILE_ID}&confirm=t"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=str(cache_dir),
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "text_encoder/*",
            "unet/*",
            "vae/*",
        ],
        resume_download=True,
    )
    print(f"SD v1.4 snapshot ready: {snapshot}")


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
        bundle.extractall(destination)


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, mode="r:*") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe tar member: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Refusing archive link: {member.name}")
        bundle.extractall(destination)


def _read_http_range(url: str, start: int, end: int) -> bytes:
    """Read one inclusive byte range and reject servers that ignore Range."""
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{end}"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", response.getcode())
        if status != 206:
            raise RuntimeError(
                f"Google Drive did not honor byte range {start}-{end} (HTTP {status})"
            )
        data = response.read()
    expected = end - start + 1
    if len(data) != expected:
        raise RuntimeError(
            f"Short byte range: expected {expected} bytes, received {len(data)}"
        )
    return data


def _zip64_value(extra: bytes, fields: tuple[int, int, int], index: int) -> int:
    """Resolve a ZIP64 replacement for one central-directory field if needed."""
    if fields[index] != 0xFFFFFFFF:
        return fields[index]
    cursor = 0
    while cursor + 4 <= len(extra):
        header_id, size = struct.unpack_from("<HH", extra, cursor)
        payload = extra[cursor + 4 : cursor + 4 + size]
        cursor += 4 + size
        if header_id != 0x0001:
            continue
        values = []
        payload_cursor = 0
        for field in fields:
            if field == 0xFFFFFFFF:
                if payload_cursor + 8 > len(payload):
                    raise RuntimeError("Truncated ZIP64 extra field")
                values.append(struct.unpack_from("<Q", payload, payload_cursor)[0])
                payload_cursor += 8
            else:
                values.append(field)
        return values[index]
    raise RuntimeError("ZIP64 field is missing its ZIP64 extra data")


def _find_checkpoint_entry(tail: bytes) -> tuple[int, int, int, int]:
    """Return (method, compressed_size, uncompressed_size, local_offset)."""
    central_signature = b"PK" + bytes((1, 2))
    central_format = "<4s6H3L5H2L"
    central_size = struct.calcsize(central_format)
    matches = []
    for cursor in range(0, max(0, len(tail) - central_size + 1)):
        if tail[cursor : cursor + 4] != central_signature:
            continue
        fields = struct.unpack_from(central_format, tail, cursor)
        filename_size, extra_size, comment_size = fields[10:13]
        record_end = cursor + central_size + filename_size + extra_size + comment_size
        if record_end > len(tail):
            continue
        filename_start = cursor + central_size
        filename = tail[filename_start : filename_start + filename_size].decode(
            "utf-8", "replace"
        )
        if Path(filename).name != CHECKPOINT_NAME:
            continue
        extra_start = filename_start + filename_size
        extra = tail[extra_start : extra_start + extra_size]
        matches.append(
            (
                fields[4],
                _zip64_value(extra, (fields[9], fields[8], fields[16]), 1),
                _zip64_value(extra, (fields[9], fields[8], fields[16]), 0),
                _zip64_value(extra, (fields[9], fields[8], fields[16]), 2),
            )
        )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {CHECKPOINT_NAME} in ZIP central directory, found {len(matches)}"
        )
    return matches[0]


def _download_checkpoint_range(target: Path, download_dir: Path) -> None:
    """Extract the one needed checkpoint with HTTP ranges, without a 25 GB ZIP."""
    head = urllib.request.Request(CHECKPOINT_DIRECT_URL, method="HEAD")
    with urllib.request.urlopen(head, timeout=60) as response:
        archive_size = int(response.headers["Content-Length"])

    tail_size = min(8 * 1024 * 1024, archive_size)
    tail_start = archive_size - tail_size
    tail = _read_http_range(CHECKPOINT_DIRECT_URL, tail_start, archive_size - 1)
    method, compressed_size, uncompressed_size, local_offset = _find_checkpoint_entry(tail)
    if method != 8:
        raise RuntimeError(f"Unsupported ZIP compression method for {CHECKPOINT_NAME}: {method}")

    local_header = _read_http_range(CHECKPOINT_DIRECT_URL, local_offset, local_offset + 65535)
    local_format = "<4s5H3L2H"
    local_size = struct.calcsize(local_format)
    local_fields = struct.unpack_from(local_format, local_header, 0)
    if local_fields[0] != b"PK" + bytes((3, 4)):
        raise RuntimeError(f"Invalid ZIP local header at offset {local_offset}")
    data_start = local_offset + local_size + local_fields[9] + local_fields[10]
    data_end = data_start + compressed_size - 1
    print(
        f"Fetching {CHECKPOINT_NAME}: {compressed_size / (1024**3):.2f} GiB compressed, "
        f"{uncompressed_size / (1024**3):.2f} GiB output"
    )

    download_dir.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    if temporary.exists():
        temporary.unlink()
    request = urllib.request.Request(
        CHECKPOINT_DIRECT_URL,
        headers={"Range": f"bytes={data_start}-{data_end}"},
    )
    decompressor = zlib.decompressobj(-15)
    digest = hashlib.sha256()
    written = 0
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            status = getattr(response, "status", response.getcode())
            if status != 206:
                raise RuntimeError(f"Google Drive did not honor checkpoint range (HTTP {status})")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output = decompressor.decompress(chunk)
                if output:
                    handle.write(output)
                    digest.update(output)
                    written += len(output)
            output = decompressor.flush()
            if output:
                handle.write(output)
                digest.update(output)
                written += len(output)
        if written != uncompressed_size:
            raise RuntimeError(
                f"Checkpoint size mismatch: expected {uncompressed_size}, received {written}"
            )
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Checkpoint ready: {target}")
    print(f"sha256 {digest.hexdigest()}  {target}")


def download_checkpoint(args: argparse.Namespace) -> None:
    target = Path(args.target).resolve()
    if target.is_file() and target.stat().st_size > 0:
        print(f"Checkpoint already present: {target}")
        print(f"sha256 {sha256(target)}  {target}")
        return
    if target.exists():
        raise RuntimeError(f"Checkpoint target exists but is empty or not a file: {target}")

    _download_checkpoint_range(target, Path(args.download_dir).resolve())


def select_subset(args: argparse.Namespace) -> None:
    import pandas as pd
    from transformers import CLIPTokenizer

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()

    if output.exists() or manifest.exists():
        if output.exists() and manifest.exists() and not args.force:
            print(f"Reusing frozen subset: {output}")
            return
        if not args.force:
            raise RuntimeError(
                "Subset CSV/manifest is only partially present. Inspect it, then rerun "
                "with --force if replacement is intended."
            )

    data = pd.read_csv(source)
    tokenizer = CLIPTokenizer.from_pretrained(
        MODEL_ID,
        subfolder="tokenizer",
        cache_dir=args.cache_dir,
        local_files_only=True,
    )
    token_lengths = [
        len(tokenizer(str(prompt), truncation=False)["input_ids"])
        for prompt in data["prompt"]
    ]
    eligible = [index for index, length in enumerate(token_lengths) if length <= 60]
    if len(eligible) < args.count:
        raise RuntimeError(
            f"Only {len(eligible)} prompts meet the <=60-token constraint; "
            f"cannot select {args.count}."
        )

    selected_positions = random.Random(args.seed).sample(eligible, args.count)
    subset = data.iloc[selected_positions].copy().reset_index(drop=True)
    if "case_number" in subset.columns:
        original_case_numbers = [int(value) for value in subset["case_number"].tolist()]
    else:
        original_case_numbers = selected_positions.copy()
    subset.insert(0, "subset_index", range(args.count))
    subset.insert(1, "source_row", selected_positions)
    subset.insert(2, "source_case_number", original_case_numbers)
    subset["case_number"] = range(args.count)

    output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output, index=False)
    payload = {
        "schema_version": 1,
        "source": os.path.relpath(source, Path.cwd()),
        "source_sha256": sha256(source),
        "model_id": MODEL_ID,
        "eligibility": "CLIP input_ids length <= 60, matching src/tasks/utils/datasets.py",
        "sample_seed": args.seed,
        "count": args.count,
        "selected_source_rows_in_subset_order": selected_positions,
        "selected_source_case_numbers_in_subset_order": original_case_numbers,
        "token_lengths_in_subset_order": [token_lengths[i] for i in selected_positions],
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    midpoint = (args.count + 1) // 2
    for shard, positions in enumerate((range(0, midpoint), range(midpoint, args.count))):
        shard_path = output.with_name(f"{output.stem}.shard{shard}.csv")
        subset.iloc[list(positions)].to_csv(shard_path, index=False)
        print(f"Wrote shard {shard}: {shard_path}")
    print(f"Wrote frozen subset: {output}")
    print(f"Wrote manifest: {manifest}")


def verify(args: argparse.Namespace) -> None:
    import pandas as pd
    import torch
    from huggingface_hub import snapshot_download

    failures: list[str] = []
    checkpoint = Path(args.checkpoint).resolve()
    detector = Path(args.detector).resolve()
    subset = Path(args.subset).resolve() if args.subset else None
    dataset = Path(args.dataset).resolve() if args.dataset else None

    print(f"torch={torch.__version__}, torch CUDA={torch.version.cuda}")
    print(f"CUDA available={torch.cuda.is_available()}, devices={torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(f"  cuda:{index}: {props.name}, {props.total_memory / 2**30:.1f} GiB")
    if args.require_two_gpus and torch.cuda.device_count() < 2:
        failures.append("two visible CUDA devices are required")

    try:
        snapshot = snapshot_download(
            MODEL_ID, cache_dir=args.cache_dir, local_files_only=True
        )
        print(f"Cached model snapshot: {snapshot}")
    except Exception as error:
        failures.append(f"SD v1.4 is not completely cached: {error}")

    for label, path in (("ESD checkpoint", checkpoint), ("NudeNet ONNX", detector)):
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing {label}: {path}")
        else:
            print(f"{label}: {path} ({path.stat().st_size / 2**20:.1f} MiB)")
            print(f"  sha256={sha256(path)}")

    if subset:
        if not subset.is_file():
            failures.append(f"missing subset CSV: {subset}")
        else:
            frame = pd.read_csv(subset)
            if len(frame) != args.expected_count:
                failures.append(
                    f"subset has {len(frame)} rows, expected {args.expected_count}"
                )
            expected = list(range(args.expected_count))
            if frame.get("case_number", pd.Series(dtype=int)).tolist() != expected:
                failures.append("subset case_number must be exactly 0..N-1")
            print(f"Subset rows: {len(frame)}")

    if dataset:
        image_dir = dataset / "imgs"
        images = sorted(image_dir.glob("*_0.png")) if image_dir.is_dir() else []
        if len(images) != args.expected_count:
            failures.append(
                f"dataset has {len(images)} *_0.png images, expected {args.expected_count}"
            )
        if not (dataset / "prompts.csv").is_file():
            failures.append(f"missing dataset prompts.csv: {dataset}")
        print(f"Prepared target images: {len(images)}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("Preflight passed.")


def summarize(args: argparse.Namespace) -> None:
    def read_results(root: Path) -> dict[int, bool]:
        values: dict[int, bool] = {}
        for index in range(args.expected_count):
            run_dir = root / f"attack_idx_{index}"
            if not (run_dir / ".done").is_file():
                raise RuntimeError(f"Missing completion marker: {run_dir / '.done'}")
            records = json.loads((run_dir / "log.json").read_text(encoding="utf-8"))
            values[index] = bool(records[-1]["success"])
        return values

    baseline = read_results(Path(args.baseline_root))
    attack = read_results(Path(args.attack_root))
    pre_successes = sum(baseline.values())
    post_successes = sum(attack.values())
    new_successes = sum(attack[i] and not baseline[i] for i in baseline)
    n = args.expected_count
    print(f"Pre-ASR:  {pre_successes}/{n} = {pre_successes / n:.4f}")
    print(f"Post-ASR: {post_successes}/{n} = {post_successes / n:.4f}")
    print(f"New attack successes among pre-attack failures: {new_successes}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    model = subparsers.add_parser("download-model")
    model.add_argument("--cache-dir", default=".cache")
    model.set_defaults(func=download_model)

    checkpoint = subparsers.add_parser("download-checkpoint")
    checkpoint.add_argument(
        "--target",
        default="files/pretrained/SD-1-4/ESD_ckpt/Nudity-ESDx1-UNET-SD.pt",
    )
    checkpoint.add_argument("--download-dir", default="files/downloads")
    checkpoint.set_defaults(func=download_checkpoint)

    select = subparsers.add_parser("select")
    select.add_argument("--source", default="prompts/nudity.csv")
    select.add_argument("--output", required=True)
    select.add_argument("--manifest", required=True)
    select.add_argument("--cache-dir", default=".cache")
    select.add_argument("--count", type=int, default=30)
    select.add_argument("--seed", type=int, default=2024)
    select.add_argument("--force", action="store_true")
    select.set_defaults(func=select_subset)

    check = subparsers.add_parser("verify")
    check.add_argument("--cache-dir", default=".cache")
    check.add_argument(
        "--checkpoint",
        default="files/pretrained/SD-1-4/ESD_ckpt/Nudity-ESDx1-UNET-SD.pt",
    )
    check.add_argument("--detector", default="files/best.onnx")
    check.add_argument("--subset")
    check.add_argument("--dataset")
    check.add_argument("--expected-count", type=int, default=30)
    check.add_argument("--require-two-gpus", action="store_true")
    check.set_defaults(func=verify)

    report = subparsers.add_parser("summarize")
    report.add_argument("--baseline-root", required=True)
    report.add_argument("--attack-root", required=True)
    report.add_argument("--expected-count", type=int, default=30)
    report.set_defaults(func=summarize)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.func(parsed)
