#!/usr/bin/env python3
"""Fetch the pinned, unmodified official MaterialX examples (no SDK build)."""
import argparse
import hashlib
from pathlib import Path
import urllib.request
import zipfile

VERSION = "1.39.5"
SHA256 = "830b33e96069a69440df175d8ab0951b292893030d825235fbb3b3f846604dda"
URL = f"https://codeload.github.com/AcademySoftwareFoundation/MaterialX/zip/refs/tags/v{VERSION}"


def fetch(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"MaterialX-{VERSION}.zip"
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        print(f"Downloading MaterialX {VERSION} from {URL}", flush=True)
        with urllib.request.urlopen(URL, timeout=120) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        partial.replace(archive)
    with archive.open("rb") as source_archive:
        digest = hashlib.file_digest(source_archive, "sha256").hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"Archive checksum mismatch: {digest}; remove {archive} and retry")
    source = destination / f"MaterialX-{VERSION}"
    # Re-extract only if the pinned tree has not already been unpacked.
    if not (source / "resources/Materials/Examples/StandardSurface/standard_surface_chess_set.mtlx").exists():
        with zipfile.ZipFile(archive) as bundle:
            for item in bundle.infolist():
                target = (destination / item.filename).resolve()
                if not target.is_relative_to(destination):
                    raise RuntimeError(f"Unsafe archive member: {item.filename}")
            bundle.extractall(destination)
    print(f"Source: {source}\nSHA256: {digest}")
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path,
                        default=Path(__file__).resolve().parents[1] / "build/deps/materialx-official")
    fetch(parser.parse_args().destination)
