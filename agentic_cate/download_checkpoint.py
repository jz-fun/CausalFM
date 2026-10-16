"""Download the pinned upstream standard-CATE checkpoint and verify its SHA-256.

Example::

    python -m agentic_cate.download_checkpoint

The artifact remains ignored by git.  This command never deserializes it.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request


UPSTREAM_COMMIT = "bb74ef729c70d274fe2bb422c3b0edff4754fa37"
CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/yccm/CausalFM-toolkit/"
    f"{UPSTREAM_COMMIT}/checkpoints/checkpoints_standard/best_model.pth"
)
CHECKPOINT_SHA256 = "9b75ebf02016a61b3a6e8fc075f52f9c56189ea5d7e7633137a5c9b70f25774e"
DEFAULT_OUTPUT = Path("checkpoints_agentic/standard.pth")
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(output: str | Path = DEFAULT_OUTPUT, *, overwrite: bool = False) -> Path:
    """Fetch pinned bytes and install atomically after checking their digest.

    An existing valid artifact is reused.  An invalid existing artifact is
    preserved unless ``overwrite=True`` is explicitly requested.  Download or
    verification failures leave the previous destination untouched.
    """
    destination = Path(output)
    if destination.exists():
        actual = sha256_file(destination)
        if actual == CHECKPOINT_SHA256:
            return destination
        if not overwrite:
            raise FileExistsError(
                f"Existing checkpoint {destination} has SHA-256 {actual}; "
                "expected " + CHECKPOINT_SHA256 + ". Pass --overwrite to replace it."
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".part",
            dir=destination.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            request = urllib.request.Request(CHECKPOINT_URL, headers={"User-Agent": "CausalFM-agentic-checkpoint/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                    stream.write(chunk)
                    digest.update(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        actual = digest.hexdigest()
        if actual != CHECKPOINT_SHA256:
            raise ValueError(f"Downloaded checkpoint SHA-256 {actual} does not match {CHECKPOINT_SHA256}")

        # Another process may have installed a file while the download ran.
        if destination.exists() and not overwrite:
            if sha256_file(destination) == CHECKPOINT_SHA256:
                return destination
            raise FileExistsError(f"Checkpoint appeared during download: {destination}; use --overwrite to replace it")
        os.replace(temporary, destination)
        temporary = None
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing artifact with the wrong SHA-256")
    args = parser.parse_args()
    path = download_checkpoint(args.output, overwrite=args.overwrite)
    print(f"Verified checkpoint: {path}")
    print(f"SHA-256: {CHECKPOINT_SHA256}")
    print(f"Upstream commit: {UPSTREAM_COMMIT}")
    print(f"Source: {CHECKPOINT_URL}")


if __name__ == "__main__":
    main()
