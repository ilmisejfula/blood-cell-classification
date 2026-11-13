"""Upload, download and list files in this project's Azure Blob container.

Run with:
    python src/storage.py upload <local_path> <blob_prefix> [--overwrite]
    python src/storage.py download <blob_prefix> <local_folder>
    python src/storage.py list [blob_prefix]

Functions upload_path, download_prefix and list_blobs can also be imported.
Each returns the number of failures (0 = success).
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

from azure.core.exceptions import AzureError, ResourceNotFoundError
from azure.storage.blob import ContainerClient, ContentSettings
from dotenv import load_dotenv
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent  # parent of src/
ENV_PATH = REPO_ROOT / ".env"
CHUNK_SIZE = 1024 * 1024  # 1 MB chunks for hashing


class StorageConfigError(RuntimeError):
    """Raised when the Azure settings are missing."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_container() -> ContainerClient:
    """Build a container client from .env. Never prints the connection string."""
    load_dotenv(ENV_PATH)
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container = os.getenv("AZURE_CONTAINER_NAME")
    missing = [
        name
        for name, value in (
            ("AZURE_STORAGE_CONNECTION_STRING", conn_str),
            ("AZURE_CONTAINER_NAME", container),
        )
        if not value
    ]
    if missing:
        raise StorageConfigError(
            f"Missing environment variable(s): {', '.join(missing)}. "
            f"Set them in {ENV_PATH} (see .env.example)."
        )
    return ContainerClient.from_connection_string(
        conn_str,
        container_name=container,
        max_single_put_size=4 * 1024 * 1024,  # files > 4 MB are uploaded in blocks
        max_block_size=4 * 1024 * 1024,       # 4 MB blocks -> progress bar updates per block
        connection_timeout=60,                # seconds to establish a connection
        read_timeout=300,                     # seconds to wait for a response
    )


def md5_file(path: Path) -> bytes:
    """MD5 digest of a file, computed in chunks."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.digest()


def md5_hex(digest) -> str:
    """Hex form of a stored MD5 (bytes/bytearray), or 'none'."""
    return bytes(digest).hex() if digest else "none"


def make_hook(bar: tqdm):
    """Adapt Azure's progress_hook(current, total) to a tqdm bar."""
    def hook(current, total):
        if total is not None and bar.total != total:
            bar.total = total
        bar.n = current
        bar.refresh()
    return hook


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_path(local_path, blob_prefix: str, overwrite: bool = False,
                container: ContainerClient | None = None) -> int:
    """Upload a file or folder (recursively). Blob name = blob_prefix + relative path."""
    container = container or get_container()
    local_path = Path(local_path)

    # Collect (file, relative path) pairs.
    if local_path.is_file():
        files = [(local_path, local_path.name)]
    elif local_path.is_dir():
        files = [(p, p.relative_to(local_path).as_posix())
                 for p in sorted(local_path.rglob("*")) if p.is_file()]
    else:
        print(f"[FAIL] Local path not found: {local_path}")
        return 1

    failures = 0
    for path, rel in files:
        blob_name = blob_prefix + rel  # forward slashes via as_posix()
        try:
            local_md5 = md5_file(path)
            blob = container.get_blob_client(blob_name)

            # Compare with an existing blob, if any.
            try:
                props = blob.get_blob_properties()
                remote_md5 = props.content_settings.content_md5
                if remote_md5 and bytes(remote_md5) == local_md5:
                    print(f"[skip] {blob_name}: already exists with matching MD5.")
                    continue
                if not overwrite:
                    print(f"[FAIL] {blob_name}: exists with different MD5 "
                          f"({md5_hex(remote_md5)}); use --overwrite to replace.")
                    failures += 1
                    continue
            except ResourceNotFoundError:
                pass  # new blob

            # Upload with the MD5 stored in the blob's content settings.
            size = path.stat().st_size
            with open(path, "rb") as f, tqdm(total=size, unit="B", unit_scale=True,
                                             desc=blob_name) as bar:
                blob.upload_blob(
                    f,
                    length=size,
                    overwrite=True,  # existence/overwrite already decided above
                    content_settings=ContentSettings(content_md5=bytearray(local_md5)),
                    progress_hook=make_hook(bar),
                    max_concurrency=4,  # upload up to 4 blocks in parallel
                )
            print(f"[ ok ] {path} -> {blob_name} (md5 {local_md5.hex()})")
        except (AzureError, OSError) as exc:
            print(f"[FAIL] {blob_name}: {exc}")
            failures += 1
    return failures


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_prefix(blob_prefix: str, local_folder,
                    container: ContainerClient | None = None) -> int:
    """Download every blob under blob_prefix, keeping the path after the prefix."""
    container = container or get_container()
    local_folder = Path(local_folder).resolve()
    failures = 0
    count = 0

    for props in container.list_blobs(name_starts_with=blob_prefix or None):
        rel = props.name[len(blob_prefix):].lstrip("/")
        if not rel or rel.endswith("/"):
            continue  # skip folder placeholders
        dest = (local_folder / rel).resolve()
        if local_folder not in dest.parents:
            print(f"[FAIL] {props.name}: unsafe path, skipped.")
            failures += 1
            continue
        count += 1
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f, tqdm(total=props.size, unit="B", unit_scale=True,
                                             desc=props.name) as bar:
                stream = container.get_blob_client(props.name).download_blob(
                    progress_hook=make_hook(bar))
                stream.readinto(f)

            # Verify against the stored MD5 when the blob has one.
            stored = props.content_settings.content_md5
            if stored:
                local_md5 = md5_file(dest)
                if local_md5 == bytes(stored):
                    print(f"[ ok ] {props.name} -> {dest} (MD5 verified)")
                else:
                    print(f"[FAIL] {props.name}: MD5 mismatch "
                          f"(got {local_md5.hex()}, stored {md5_hex(stored)})")
                    failures += 1
            else:
                print(f"[ ok ] {props.name} -> {dest} (no stored MD5, not verified)")
        except (AzureError, OSError) as exc:
            print(f"[FAIL] {props.name}: {exc}")
            failures += 1

    if count == 0:
        print(f"[FAIL] No blobs found under prefix '{blob_prefix}'.")
        failures += 1
    return failures


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

def list_blobs(blob_prefix: str = "", container: ContainerClient | None = None) -> int:
    """Print name, size (bytes and MB) and stored MD5 for each blob under the prefix."""
    container = container or get_container()
    total_bytes = 0
    count = 0
    print(f"{'name':<50} {'bytes':>14} {'MB':>10}  md5")
    for props in container.list_blobs(name_starts_with=blob_prefix or None):
        count += 1
        total_bytes += props.size
        print(f"{props.name:<50} {props.size:>14} {props.size / 1_000_000:>10.2f}  "
              f"{md5_hex(props.content_settings.content_md5)}")
    print(f"\nTotal: {count} blob(s), {total_bytes} bytes ({total_bytes / 1_000_000:.2f} MB)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Azure Blob storage helper.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_up = sub.add_parser("upload", help="Upload a file or folder.")
    p_up.add_argument("local_path")
    p_up.add_argument("blob_prefix", help="e.g. raw/ (include the trailing slash)")
    p_up.add_argument("--overwrite", action="store_true",
                      help="Replace existing blobs whose MD5 differs.")

    p_down = sub.add_parser("download", help="Download all blobs under a prefix.")
    p_down.add_argument("blob_prefix")
    p_down.add_argument("local_folder")

    p_list = sub.add_parser("list", help="List blobs.")
    p_list.add_argument("blob_prefix", nargs="?", default="")

    args = parser.parse_args(argv)
    try:
        if args.command == "upload":
            failures = upload_path(args.local_path, args.blob_prefix, args.overwrite)
        elif args.command == "download":
            failures = download_prefix(args.blob_prefix, args.local_folder)
        else:
            failures = list_blobs(args.blob_prefix)
    except StorageConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1
    except (AzureError, ValueError) as exc:
        # ValueError covers a malformed connection string; its message does not include it.
        print(f"Azure error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if failures:
        print(f"\n{failures} failure(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
