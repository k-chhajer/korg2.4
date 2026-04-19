from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

from committee_llm.benchmark_data import convert_2wikimultihop


DEFAULT_DATA_URL = "https://www.dropbox.com/s/npidmtadreo6df2/data.zip?dl=1"


def _find_split_file(root: Path, split: str) -> Path:
    candidates = sorted(root.rglob(f"{split}.json"))
    if not candidates:
        raise FileNotFoundError(f"Could not find {split}.json under {root}")

    preferred = [
        path
        for path in candidates
        if "wikimultihop" in str(path).lower() or path.parent.name.lower() in {"data", "wikimultihop"}
    ]
    return preferred[0] if preferred else candidates[0]


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"using existing archive: {target}")
        return
    print(f"downloading {url}")
    urlretrieve(url, target)
    print(f"downloaded archive: {target}")


def _extract(archive: Path, extract_dir: Path) -> None:
    marker = extract_dir / ".extracted"
    if marker.exists():
        print(f"using existing extraction: {extract_dir}")
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"extracting {archive} -> {extract_dir}")
    with ZipFile(archive) as handle:
        handle.extractall(extract_dir)
    marker.write_text("ok\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and convert 2WikiMultihopQA for Arch 2 RL training.")
    parser.add_argument("--url", default=DEFAULT_DATA_URL, help="Dataset zip URL.")
    parser.add_argument("--archive", default="../evals/data/raw/2wikimultihop/data.zip", help="Local zip path.")
    parser.add_argument("--extract-dir", default="../evals/data/raw/2wikimultihop/extracted", help="Extraction dir.")
    parser.add_argument("--output-dir", default="../evals/data/benchmarks", help="Converted JSONL output dir.")
    parser.add_argument("--train-limit", type=int, help="Optional max train examples to convert.")
    parser.add_argument("--dev-limit", type=int, help="Optional max dev examples to convert.")
    parser.add_argument("--skip-download", action="store_true", help="Use an existing archive without downloading.")
    parser.add_argument("--skip-extract", action="store_true", help="Use an existing extracted directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    base_dir = Path(__file__).resolve().parents[1]
    archive = Path(args.archive)
    extract_dir = Path(args.extract_dir)
    output_dir = Path(args.output_dir)

    if not archive.is_absolute():
        archive = (base_dir / archive).resolve()
    if not extract_dir.is_absolute():
        extract_dir = (base_dir / extract_dir).resolve()
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()

    if not args.skip_download:
        _download(args.url, archive)
    elif not archive.exists():
        raise FileNotFoundError(f"--skip-download was set but archive does not exist: {archive}")

    if not args.skip_extract:
        _extract(archive, extract_dir)

    train_json = _find_split_file(extract_dir, "train")
    dev_json = _find_split_file(extract_dir, "dev")
    print(f"train source: {train_json}")
    print(f"dev source:   {dev_json}")

    train_out = output_dir / "2wikimultihop_train.jsonl"
    dev_out = output_dir / "2wikimultihop_dev.jsonl"
    train_count = convert_2wikimultihop(train_json, train_out, limit=args.train_limit, split="train")
    dev_count = convert_2wikimultihop(dev_json, dev_out, limit=args.dev_limit, split="dev")

    print(f"wrote {train_count} train tasks to {train_out}")
    print(f"wrote {dev_count} dev tasks to {dev_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
