"""Generate server/cuda_manifest.json from packaging/cuda_allowlist.txt.

Run against a venv that has the NVIDIA wheels installed. The manifest is what the runtime
uses to decide whether a downloaded payload is complete, and what the downloader verifies
each file against, so it has to be regenerated whenever the pinned CUDA versions move.

It is written into server/ rather than packaging/ because stage-backend.ps1 copies only
server/ and ui/ into the package: a manifest left in packaging/ would not exist at runtime.

    python packaging/make_cuda_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import lzma
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST = ROOT / "packaging" / "cuda_allowlist.txt"
OUT = ROOT / "server" / "cuda_manifest.json"
ASSETS = ROOT / "packaging" / "cuda-assets"

# The versions the manifest describes. Recorded so a mismatch between the manifest and the
# wheels actually installed is visible in the file rather than inferred from a digest.
CUBLAS_VERSION = "12.9.2.10"
NVRTC_VERSION = "12.9.86"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compress(args: tuple[str, str, str]) -> tuple[str, int, str, int, str]:
    """Compress one DLL to .xz and return both sides' size and digest.

    Run in a worker process: LZMA is single-threaded and cublasLt alone takes about two and
    a half minutes, so compressing the four in parallel turns a six minute build step into
    roughly the time of the longest file.

    preset 6 rather than 9. Measured on the real payload, 9 buys a few more megabytes for a
    large increase in build time, and the transfer is dominated by cublasLt either way.
    """
    rel, src_s, dst_s = args
    src, dst = Path(src_s), Path(dst_s)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fin, lzma.open(dst, "wb", preset=6) as fout:
        shutil.copyfileobj(fin, fout, 1 << 20)
    return rel, src.stat().st_size, sha256(src), dst.stat().st_size, sha256(dst)


def main() -> int:
    nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nvidia.is_dir():
        print(f"[error] no NVIDIA payload at {nvidia}; "
              "run this from a venv with the CUDA wheels installed", file=sys.stderr)
        return 1

    jobs = []
    for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        rel = line.strip()
        if not rel or rel.startswith("#"):
            continue
        src = nvidia / rel
        if not src.is_file():
            print(f"[error] allow-listed file missing: {src}", file=sys.stderr)
            return 1
        jobs.append((rel, str(src), str(ASSETS / (rel.rsplit("/", 1)[-1] + ".xz"))))

    ASSETS.mkdir(parents=True, exist_ok=True)
    print(f"compressing {len(jobs)} DLLs in parallel (a few minutes) ...")
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        results = list(pool.map(compress, jobs))

    order = {rel: i for i, (rel, _, _) in enumerate(jobs)}
    results.sort(key=lambda r: order[r[0]])

    entries = []
    for rel, size, digest, xz_size, xz_digest in results:
        entries.append({
            "path": rel,
            "size": size,
            "sha256": digest,
            "xz_size": xz_size,
            "xz_sha256": xz_digest,
        })
        print(f"  {rel.rsplit('/', 1)[-1]:28} {size / 1e6:8.1f} MB -> "
              f"{xz_size / 1e6:7.1f} MB  ({xz_size / size:.1%})")

    OUT.write_text(json.dumps({
        "release_tag": "cuda-12.9.2",
        "cublas_version": CUBLAS_VERSION,
        "nvrtc_version": NVRTC_VERSION,
        "files": entries,
    }, indent=2) + "\n", encoding="utf-8")

    total = sum(e["size"] for e in entries)
    xz_total = sum(e["xz_size"] for e in entries)
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({len(entries)} files)")
    print(f"  on disk        {total / 1e6:8.1f} MB")
    print(f"  over the wire  {xz_total / 1e6:8.1f} MB")
    print(f"  assets in      {ASSETS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
