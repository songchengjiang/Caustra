#!/usr/bin/env python3
"""Fetch only San Miguel from the pinned official PBRT scene repository."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from urllib.parse import quote

COMMIT = "30cf4a0346ae5a80a2d7a530a3ef7d0fa4f70572"
REPOSITORY = "mmp/pbrt-v4-scenes"
API = "https://api.github.com/repos/" + REPOSITORY
RAW = "https://raw.githubusercontent.com/" + REPOSITORY + "/" + COMMIT + "/"


def request(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Caustra-SanMiguel-asset-fetch"}), timeout=120) as response:
                return response.read()
        except OSError:
            if attempt == 4:
                raise
            time.sleep(1 + attempt)


def git_hash(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/deps/pbrt-v4-scenes"))
    parser.add_argument("--workers", type=int, default=8)
    options = parser.parse_args()
    if not 1 <= options.workers <= 16:
        parser.error("workers must be in 1..16")
    root = json.loads(request(API + "/git/trees/" + COMMIT))
    subtree = next(entry["sha"] for entry in root["tree"] if entry["path"] == "sanmiguel" and entry["type"] == "tree")
    tree = json.loads(request(API + "/git/trees/" + subtree + "?recursive=1"))
    if tree.get("truncated"):
        raise RuntimeError("GitHub returned a truncated scene inventory")
    files = [dict(entry, path="sanmiguel/" + entry["path"]) for entry in tree["tree"] if entry["type"] == "blob"]
    files += [entry for entry in root["tree"] if entry["type"] == "blob" and entry["path"] in ("README.md", "LICENSE", "LICENSE.txt")]
    print("Pinned PBRT %s: %d files, %.1f MiB" % (COMMIT, len(files), sum(f.get("size", 0) for f in files) / 2**20), flush=True)
    destination = options.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    def fetch(entry):
        target = (destination / entry["path"]).resolve()
        if not target.is_relative_to(destination):
            raise RuntimeError("Unsafe source path")
        data = target.read_bytes() if target.is_file() else None
        if data is None or git_hash(data) != entry["sha"]:
            data = request(RAW + quote(entry["path"], safe="/"))
            if git_hash(data) != entry["sha"]:
                raise RuntimeError("Source hash mismatch: " + entry["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            temporary.write_bytes(data)
            temporary.replace(target)
        return {"path": entry["path"], "bytes": len(data), "git_blob": entry["sha"], "sha256": hashlib.sha256(data).hexdigest()}

    inventory = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=options.workers) as executor:
        futures = {executor.submit(fetch, entry): entry["path"] for entry in files}
        for future in concurrent.futures.as_completed(futures):
            inventory.append(future.result())
            if len(inventory) % 50 == 0 or len(inventory) == len(files):
                print("Verified %d/%d files" % (len(inventory), len(files)), flush=True)
    manifest = {"repository": "https://github.com/" + REPOSITORY, "commit": COMMIT,
                "author": "Guillermo M. Leal Llaguno", "files": sorted(inventory, key=lambda f: f["path"])}
    (destination / "sanmiguel-source.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Source: " + str(destination / "sanmiguel"), flush=True)


if __name__ == "__main__":
    main()
