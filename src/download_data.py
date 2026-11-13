"""Download the official BloodMNIST files from Zenodo into data/raw/ and verify them by MD5.

Run with: python src/download_data.py
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

from tqdm import tqdm

# (URL, local file name, expected MD5)
FILES = [
    (
        "https://zenodo.org/records/10519652/files/bloodmnist.npz?download=1",
        "bloodmnist.npz",
        "7053d0359d879ad8a5505303e11de1dc",
    ),
    (
        "https://zenodo.org/records/10519652/files/bloodmnist_64.npz?download=1",
        "bloodmnist_64.npz",
        "2b94928a2ae4916078ca51e05b6b800b",
    ),
    (
        "https://zenodo.org/records/10519652/files/bloodmnist_128.npz?download=1",
        "bloodmnist_128.npz",
        "adace1e0ed228fccda1f39692059dd4c",
    ),
]

CHUNK_SIZE = 1024 * 1024  # 1 MB read/write chunks
REPO_ROOT = Path(__file__).resolve().parent.parent  # parent of src/
RAW_DIR = REPO_ROOT / "data" / "raw"


def md5_of(path: Path) -> str:
    """Compute the MD5 of a file in chunks, without loading it all into memory."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Stream a URL to dest in 1 MB chunks, showing a tqdm progress bar."""
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
        total = response.headers.get("Content-Length")
        total = int(total) if total else None
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
                bar.update(len(chunk))


def process(url: str, name: str, expected: str) -> dict:
    """Make sure one file is present and verified; return a row for the summary."""
    final_path = RAW_DIR / name
    part_path = RAW_DIR / f"{name}.part"

    # Skip the download if a verified copy already exists.
    if final_path.exists():
        computed = md5_of(final_path)
        if computed == expected:
            print(f"[skip] {name} already exists and MD5 matches.")
            return row(name, final_path, computed, expected)
        print(f"[redo] {name} exists but MD5 differs ({computed}); downloading again.")

    # Download to <name>.part, then verify before renaming.
    print(f"[get ] {name} <- {url}")
    try:
        download(url, part_path)
    except Exception as exc:  # network/HTTP/disk errors
        print(f"[FAIL] {name}: download error: {exc}")
        return row(name, part_path, "-", expected, failed=True)

    computed = md5_of(part_path)
    if computed == expected:
        part_path.replace(final_path)
        print(f"[ ok ] {name} verified.")
        return row(name, final_path, computed, expected)

    print(f"[FAIL] {name}: MD5 mismatch (got {computed}, expected {expected}); kept {part_path.name}.")
    return row(name, part_path, computed, expected)


def row(name: str, path: Path, computed: str, expected: str, failed: bool = False) -> dict:
    """Build one summary-table row."""
    size = path.stat().st_size if path.exists() else 0
    return {
        "name": name,
        "bytes": size,
        "mb": size / 1_000_000,  # 1 MB = 1,000,000 bytes
        "computed": computed,
        "expected": expected,
        "status": "OK" if (not failed and computed == expected) else "FAIL",
    }


def print_summary(rows: list[dict]) -> None:
    """Print a fixed-width summary table."""
    header = f"{'file':<20} {'bytes':>12} {'MB':>9}  {'computed MD5':<32}  {'expected MD5':<32}  status"
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<20} {r['bytes']:>12} {r['mb']:>9.2f}  "
            f"{r['computed']:<32}  {r['expected']:<32}  {r['status']}"
        )


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = [process(url, name, md5) for url, name, md5 in FILES]
    print_summary(rows)
    return 1 if any(r["status"] == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
