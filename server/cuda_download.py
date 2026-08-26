"""Fetch the CUDA payload on demand.

The four allow-listed DLLs are 828 MB uncompressed, which was 61% of the MSIX. They are
published as individual .xz assets on a pinned GitHub release and installed into LocalAppData
the first time a machine that can use them asks for a model that needs them.

Four separate assets rather than one archive, for two reasons. Resume granularity: cublasLt
is 77% of the payload, so a single archive means one failure late in a 455 MB transfer throws
away everything. And integrity per file, which is what ``cuda_setup.payload_present`` checks
against.

Publication is atomic. Files are decompressed into a staging directory and only moved into
place once every one of them has arrived and matched its digest, after which the sentinel is
written. A partial payload must never be visible to the resolver: CTranslate2 loads cuBLAS
lazily, so an incomplete install is not caught at startup, it is caught at the first thing
somebody says.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import shutil
from pathlib import Path
from typing import Callable

from .cuda_setup import MANIFEST, SENTINEL, download_root, manifest

def _release_tag() -> str:
    """The release the payload is pinned to, read from the manifest.

    Single source of truth deliberately. This was previously a constant here *and* a field
    the manifest generator wrote, with nothing keeping them equal: bumping one and not the
    other would fetch last release's assets, mismatch every digest the new manifest expects,
    fail the checksum, and leave the machine permanently and silently on the processor.
    """
    try:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        tag = data.get("release_tag")
        if tag:
            return str(tag)
    except (OSError, ValueError):
        pass
    return "cuda-12.9.2"


RELEASE_TAG = _release_tag()
ASSET_BASE = f"https://github.com/desigrit/sunno/releases/download/{RELEASE_TAG}"

CHUNK = 1 << 20
_TIMEOUT = 60.0


def asset_name(rel_path: str) -> str:
    """`cublas/bin/cublas64_12.dll` -> `cublas64_12.dll.xz`, the published asset name."""
    return rel_path.rsplit("/", 1)[-1] + ".xz"


def compressed_total() -> int:
    """Bytes actually transferred, for a progress denominator that does not lie.

    The manifest records uncompressed sizes because that is what the resolver checks on
    disk. Quoting those to a progress bar would make a 455 MB download claim 828 MB and
    finish at 55%.
    """
    return sum(f.get("xz_size", 0) for f in manifest())


def already_installed() -> bool:
    from .cuda_setup import payload_present

    return payload_present(download_root())


def _fetch(client, url: str, dest: Path, expect_sha: str,
           on_bytes: Callable[[int], None]) -> None:
    """Download one asset with Range resume, decompressing as it lands.

    The .part file holds compressed bytes so a resumed transfer can pick up at a byte
    offset; decompression happens once, at the end, when the whole asset is known good.
    """
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}

    with client.stream("GET", url, headers=headers, timeout=_TIMEOUT) as r:
        if have and r.status_code == 200:
            # Server ignored the range; start over rather than corrupt the file.
            have = 0
            part.unlink(missing_ok=True)
        elif r.status_code not in (200, 206):
            raise RuntimeError(f"{url} returned HTTP {r.status_code}")
        if have:
            on_bytes(have)
        mode = "ab" if have else "wb"
        with part.open(mode) as fh:
            for chunk in r.iter_bytes(CHUNK):
                fh.write(chunk)
                on_bytes(len(chunk))

    # Verify the compressed asset before spending time decompressing it.
    h = hashlib.sha256()
    with part.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    if h.hexdigest() != expect_sha:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{url} failed its checksum; refusing to install it")

    dest.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(part, "rb") as src, dest.open("wb") as out:        shutil.copyfileobj(src, out, CHUNK)
    part.unlink(missing_ok=True)


def download_payload(on_progress: Callable[[int, int], None] | None = None) -> None:
    """Install the CUDA payload, or raise.

    Args:
        on_progress: called with (bytes_done, bytes_total) in compressed bytes.

    Raises on any failure. Callers fall back to CPU and say so; they must not treat a
    failure here as fatal, because captions on the processor are the whole reason this is
    allowed to be optional.
    """
    import httpx

    files = manifest()
    if not files:
        raise RuntimeError("no CUDA manifest; this build cannot install GPU support")

    root = download_root()
    staging = root.parent / "cuda-staging"
    staging.mkdir(parents=True, exist_ok=True)

    total = compressed_total()
    done = 0

    def bump(n: int) -> None:
        nonlocal done
        done += n
        if on_progress:
            on_progress(min(done, total), total)

    ok = False
    try:
        with httpx.Client(follow_redirects=True, timeout=_TIMEOUT) as client:
            for f in files:
                dest = staging / f["path"]
                # Already decompressed by an earlier attempt. Counting its bytes rather than
                # refetching is the other half of making resume real: without this, a failure
                # on the last asset would re-download the three that had already succeeded.
                if dest.is_file() and dest.stat().st_size == f["size"]:
                    bump(f["xz_size"])
                    continue
                url = f"{ASSET_BASE}/{asset_name(f['path'])}"
                _fetch(client, url, dest, f["xz_sha256"], bump)

        # Everything landed. Check sizes against the manifest before publishing, so a
        # truncated decompression cannot become a visible payload.
        for f in files:
            p = staging / f["path"]
            if not p.is_file() or p.stat().st_size != f["size"]:
                raise RuntimeError(f"{f['path']} is the wrong size after decompression")

        # Move the old payload aside before putting the new one in place, rather than
        # deleting it first. rmtree cannot remove a DLL another process has loaded, and with
        # ignore_errors it fails quietly and leaves the directory behind, so the rename then
        # raises and the finally discards a download that had completely succeeded. The
        # machine was left with a half-gutted payload, no sentinel, and nothing to show for
        # the transfer. Renaming first means the worst case is a stale directory to clean up.
        old = None
        if root.exists():
            old = root.parent / "cuda-old"
            shutil.rmtree(old, ignore_errors=True)
            root.rename(old)
        root.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(root)
        (root / SENTINEL).write_text(RELEASE_TAG, encoding="utf-8")
        ok = True
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
    finally:
        # Only cleared on success. The .part files live in here, and deleting them on failure
        # is what made the Range resume above dead code: every retry restarted all 455 MB.
        # Leaving them means an interrupted download resumes where it stopped.
        if ok and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
