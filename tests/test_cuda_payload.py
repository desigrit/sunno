"""Checks for the on-demand CUDA payload.

The payload used to ship inside the MSIX, where stage-backend.ps1 guaranteed it was
all-or-nothing. It is now downloaded, so completeness has to be re-established at runtime and
these checks are what hold that line.

Why this file is worth its length: nothing about a missing CUDA payload fails loudly.
CTranslate2 resolves cuBLAS lazily, so a machine with a half-installed payload constructs a
CUDA engine successfully and only fails at the first decode. The checks below are the reason
that never reaches a user.

    python tests/test_cuda_payload.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tests._isolate  # noqa: F401,E402

from server import cuda_setup, models  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

failures: list[str] = []
checks = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if not ok:
        failures.append(f"{label}{(': ' + detail) if detail else ''}")


def section(name: str) -> None:
    print(f"\n-- {name}")


# ---------------------------------------------------------------- manifest
section("manifest")

manifest = cuda_setup.manifest()
check("manifest is present", bool(manifest),
      "server/cuda_manifest.json missing; run packaging/make_cuda_manifest.py")

allowlist = [
    ln.strip() for ln in (REPO / "packaging" / "cuda_allowlist.txt")
    .read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")
]
check("manifest matches the allow-list",
      sorted(f["path"] for f in manifest) == sorted(allowlist),
      f"{[f['path'] for f in manifest]} vs {allowlist}")

for f in manifest:
    check(f"{f['path']} has a size", f.get("size", 0) > 0)
    check(f"{f['path']} has a sha256", len(f.get("sha256", "")) == 64)
    # The compressed digest is what the downloader verifies before it unpacks anything, so a
    # manifest without it cannot check what it fetched.
    check(f"{f['path']} has an xz sha256", len(f.get("xz_sha256", "")) == 64)
    check(f"{f['path']} has an xz size", f.get("xz_size", 0) > 0)
    check(f"{f['path']} compresses smaller", f.get("xz_size", 0) < f.get("size", 1))

# cuDNN ships inside the ctranslate2 wheel, not the nvidia payload. It appearing here would
# mean downloading a file that is already installed and then looking for it in the wrong
# place.
check("cudnn is not in the payload",
      not any("cudnn" in f["path"].lower() for f in manifest))


# ------------------------------------------------------- completeness rules
section("completeness")


def fake_payload(tmp: Path, *, complete: bool = True, sentinel: bool = True,
                 truncate: str | None = None) -> Path:
    root = tmp / "cuda" / "nvidia"
    files = manifest if complete else manifest[:-1]
    for f in files:
        p = root / f["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        size = f["size"] - 1 if truncate == f["path"] else f["size"]
        with p.open("wb") as fh:
            fh.truncate(size)
    if sentinel:
        (root / cuda_setup.SENTINEL).write_text("test", encoding="utf-8")
    return root


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    real_dl = cuda_setup.download_root

    root = fake_payload(tmp)
    cuda_setup.download_root = lambda: root
    check("a complete payload is accepted", cuda_setup.payload_present(root))

    # The failure this whole design exists to prevent. The check it replaced tested only that
    # a <pkg>/bin directory existed, which a partial download satisfies immediately.
    root2 = fake_payload(tmp / "partial", complete=False)
    cuda_setup.download_root = lambda: root2
    check("a partial payload is rejected", not cuda_setup.payload_present(root2))

    root3 = fake_payload(tmp / "trunc", truncate=manifest[0]["path"])
    cuda_setup.download_root = lambda: root3
    check("a truncated file is rejected", not cuda_setup.payload_present(root3))

    root4 = fake_payload(tmp / "nosent", sentinel=False)
    cuda_setup.download_root = lambda: root4
    check("a payload with no sentinel is rejected", not cuda_setup.payload_present(root4))

    cuda_setup.download_root = real_dl


# -------------------------------------------------------- registration memo
section("registration")

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    root = fake_payload(tmp)
    real_candidates = cuda_setup.candidate_roots
    real_dl = cuda_setup.download_root
    saved_memo = cuda_setup._registered_root

    # The exact shape of the bug this replaced: the old code set its "registered" flag inside
    # the failure branch, so an import-time call that found nothing made every later call
    # return early without ever calling os.add_dll_directory.
    cuda_setup._registered_root = None
    cuda_setup.candidate_roots = lambda: [Path("Z:/definitely-absent")]
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cuda_setup.register_cuda_dlls()
    check("a failed lookup does not latch the memo", cuda_setup._registered_root is None)

    raised = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cuda_setup.register_cuda_dlls(required=True)
    except RuntimeError:
        raised = True
    check("required=True raises when absent", raised)
    check("a raise still does not latch the memo", cuda_setup._registered_root is None)

    cuda_setup.download_root = lambda: root
    cuda_setup.candidate_roots = lambda: [root]
    added = cuda_setup.register_cuda_dlls()
    check("registration works once the payload arrives", bool(added),
          "this is the regression that silently disabled the GPU")
    check("registers one dir per package", len(added) == 2, str(added))
    check("memo now holds the root", cuda_setup._registered_root == root)
    check("a repeat call is a no-op", cuda_setup.register_cuda_dlls() == [])

    cuda_setup.candidate_roots = real_candidates
    cuda_setup.download_root = real_dl
    cuda_setup._registered_root = saved_memo


# ------------------------------------------------------------- catalogue
section("catalogue")

catalog = models.catalog_with_status("cpu")
check("every entry declares uses_gpu", all("uses_gpu" in e for e in catalog))

streaming = [e for e in catalog if e["id"].startswith("stream-")]
check("streaming models exist in the catalogue", len(streaming) >= 2)
check("no streaming model uses the GPU", all(not e["uses_gpu"] for e in streaming),
      "picking one of these must never trigger a 455 MB download")
whisper = [e for e in catalog if not e["id"].startswith("stream-")]
check("every Whisper model uses the GPU", all(e["uses_gpu"] for e in whisper))

# uses_gpu has to follow is_stream_model, not a string prefix. Nothing else in the codebase
# treats "stream-" as meaningful, and a second parallel rule would drift from the first.
check("uses_gpu tracks is_stream_model",
      all(e["uses_gpu"] != models.is_stream_model(e["id"]) for e in catalog))


# ----------------------------------------------------------- device resolve
section("device resolution")

from server import hardware  # noqa: E402

real_has = hardware.has_cuda
real_ready = hardware.payload_ready
try:
    hardware.has_cuda = lambda: True
    hardware.payload_ready = lambda: False
    check("no payload means the processor",
          hardware.resolve_device("auto") == "cpu",
          "resolve_device promises a device that will really load")
    # Explicit --compute-device cuda cannot be honoured either: app.py's strict check would
    # send it back to the processor moments later regardless of what is returned here.
    check("an explicit cuda request is still honest",
          hardware.resolve_device("cuda") == "cpu")

    hardware.payload_ready = lambda: True
    check("a complete payload means cuda", hardware.resolve_device("auto") == "cuda")

    hardware.has_cuda = lambda: False
    check("no GPU means the processor even with a payload",
          hardware.resolve_device("auto") == "cpu")
    check("forcing cpu is always respected", hardware.resolve_device("cpu") == "cpu")
finally:
    hardware.has_cuda = real_has
    hardware.payload_ready = real_ready


# ------------------------------------------------------------- packaging
section("packaging")

stage = (REPO / "packaging" / "stage-backend.ps1").read_text(encoding="utf-8")
# Asserted against what the script does, not what its comments say. An earlier version of
# this check searched for the allow-list filename anywhere in the file and tripped over the
# comment explaining why the payload is no longer staged, which is prose worth keeping.
check("staging does not read the allow-list", "Get-Content $allow" not in stage,
      "the payload must not go back into the package")
check("staging does not copy DLLs into the nvidia tree",
      "nvidia\\$($rel" not in stage)
check("staging still excludes the nvidia tree", '"nvidia"' in stage)
check("staging requires the manifest", "cuda_manifest.json" in stage)
check("the SkipCuda switch is gone", "SkipCuda" not in stage,
      "a parameter that no longer does anything is a trap for the next caller")

# The manifest has to be inside server/, because staging copies only server/ and ui/.
check("the manifest ships", (REPO / "server" / "cuda_manifest.json").is_file())


# ------------------------------------------------------------------ docs
section("documentation")

privacy = (REPO / "PRIVACY.md").read_text(encoding="utf-8")
readme = (REPO / "README.md").read_text(encoding="utf-8")
notices = (REPO / "THIRD-PARTY-NOTICES.md").read_text(encoding="utf-8")

# These two sentences were true when the payload shipped in the package and became false the
# moment it did not. A privacy promise that quietly stops being accurate is worse than one
# that was never made.
check("privacy no longer claims a single network operation",
      "exactly one network operation" not in privacy)
check("readme no longer claims one step only",
      "that one step only" not in readme)
check("privacy names the second download",
      "GPU support libraries" in privacy and "NVIDIA" in privacy)
check("privacy still promises nothing is uploaded",
      "never your audio or your captions" in privacy)
check("the storage table lists the cuda directory", "`cuda\\`" in privacy)
check("the storage table lists stream models", "`stream-models\\`" in privacy)
check("notices say CUDA is not in the package",
      "Not in the package" in notices)

# An offline machine must still be described accurately: it captions, just on the processor.
check("privacy explains the offline case",
      "If you never connect at all" in privacy)

# The setup screen is the one surface where someone commits to the download, so its own copy
# has to match the promise the docs make. It used to say the model "is downloaded once and
# reused", which stopped being the whole truth the moment a second thing could download.
setup_xaml = (REPO / "app" / "MainWindow.xaml").read_text(encoding="utf-8")
check("the setup screen no longer claims a single download",
      "downloaded once and reused" not in setup_xaml)
check("the setup screen has a slot for the GPU note",
      'x:Name="GpuPayloadNote"' in setup_xaml)

# The note is built from the backend's figure rather than a literal, so it cannot drift away
# from what is actually published on the release.
window_cs = (REPO / "app" / "MainWindow.xaml.cs").read_text(encoding="utf-8")
check("the note's size comes from the backend",
      "about {rounded} MB" in window_cs and "_gpuPayloadMb / 50 * 50" in window_cs,
      "a hardcoded figure drifts the moment the pinned CUDA version moves")
# The repair path depends on this: a model that is already on disk must still enter the
# download branch when the libraries are missing, or choosing it fetches nothing.
check("the payload check precedes the availability branch",
      window_cs.index("var needsPayload") < window_cs.index("if (!row.Available || needsPayload)"))
check("an already-downloaded model still repairs",
      "!row.Available || needsPayload" in window_cs)
# Deliberately NOT reachable for the already-selected model. A RadioButton raises no Checked
# event when the checked one is clicked, so such a branch could only ever fire from a late
# container realization, starting an unrequested 477 MB download and a backend restart.
check("selecting the current model is still an early return",
      "if (row is null || id == _settings.Model) return;" in window_cs)
# The session that downloads the payload has to use it. settings.device is fixed before any
# download happens, so without this the first-run GPU user pays for 477 MB and then spends
# the whole session on the processor.
app_py = (REPO / "server" / "app.py").read_text(encoding="utf-8")
check("the device is re-resolved after the download",
      "GPU libraries are now installed" in app_py)
check("the re-resolve only ever upgrades",
      'if settings.device != "cuda":' in app_py)


# ---------------------------------------------------------------- download
section("downloader")

from server import cuda_download  # noqa: E402

check("the release tag is pinned", cuda_download.RELEASE_TAG.startswith("cuda-"))
check("assets come from the pinned tag",
      cuda_download.RELEASE_TAG in cuda_download.ASSET_BASE)
# The tag lived in two places with nothing keeping them equal. Bumping one and not the other
# fetches the previous release's assets, mismatches every digest, and leaves the machine
# silently on the processor for good.
_manifest_tag = json.loads((REPO / "server" / "cuda_manifest.json")
                           .read_text(encoding="utf-8")).get("release_tag")
check("the downloader reads the tag from the manifest",
      cuda_download.RELEASE_TAG == _manifest_tag,
      f"{cuda_download.RELEASE_TAG} vs {_manifest_tag}")
check("asset names derive from the manifest",
      cuda_download.asset_name("cublas/bin/cublas64_12.dll") == "cublas64_12.dll.xz")
# The progress denominator must be compressed bytes; quoting the on-disk total would make a
# 455 MB transfer claim 828 MB and stop at 55%.
check("the progress total is compressed bytes",
      cuda_download.compressed_total() == sum(f["xz_size"] for f in manifest))
check("compressed total is smaller than on-disk",
      cuda_download.compressed_total() < sum(f["size"] for f in manifest))


# ------------------------------------------------- downloader, exercised for real
section("downloader behaviour")

import lzma  # noqa: E402


class FakeResponse:
    def __init__(self, body: bytes, status: int = 206):
        self._body = body
        self.status_code = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_bytes(self, n):
        for i in range(0, len(self._body), n):
            yield self._body[i:i + n]


class FakeClient:
    """Serves the manifest's assets from memory, optionally failing partway.

    Exists because the real code path had no coverage at all: the checks above only tested
    constants and a string transform, on the newest and riskiest module in the change.
    """

    def __init__(self, blobs, fail_on=None, record=None):
        self.blobs, self.fail_on, self.record = blobs, fail_on, record if record is not None else []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def stream(self, method, url, headers=None, timeout=None):
        name = url.rsplit("/", 1)[-1]
        start = 0
        rng = (headers or {}).get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
        self.record.append((name, start))
        if name == self.fail_on:
            raise OSError("simulated network failure")
        return FakeResponse(self.blobs[name][start:], 206 if start else 200)


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    real_dl = cuda_download.download_root
    real_cs_dl = cuda_setup.download_root
    root = tmp / "cuda" / "nvidia"
    cuda_download.download_root = lambda: root
    cuda_setup.download_root = lambda: root

    # Build compressed assets whose digests match a manifest we substitute in.
    small = [
        {"path": "cublas/bin/a.dll", "size": 2048},
        {"path": "cuda_nvrtc/bin/b.dll", "size": 1024},
    ]
    blobs = {}
    for f in small:
        raw = bytes(f["size"])
        xz = lzma.compress(raw, preset=1)
        blobs[f["path"].rsplit("/", 1)[-1] + ".xz"] = xz
        f["sha256"] = hashlib.sha256(raw).hexdigest()
        f["xz_sha256"] = hashlib.sha256(xz).hexdigest()
        f["xz_size"] = len(xz)

    real_manifest = cuda_download.manifest
    cuda_download.manifest = lambda: small
    cuda_setup.MANIFEST_OVERRIDE = small
    real_cs_manifest = cuda_setup.manifest
    cuda_setup.manifest = lambda: small

    import httpx as _httpx  # noqa: E402

    real_client = _httpx.Client
    seen: list[tuple[str, int]] = []

    # 1. Happy path publishes a complete payload.
    _httpx.Client = lambda **kw: FakeClient(blobs, record=seen)
    pcts = []
    cuda_download.download_payload(lambda d, t: pcts.append((d, t)))
    check("a clean download publishes", cuda_setup.payload_present(root))
    check("the sentinel is written", (root / cuda_setup.SENTINEL).exists())
    check("no staging is left behind", not (root.parent / "cuda-staging").exists())
    check("progress is monotonic", all(a[0] <= b[0] for a, b in zip(pcts, pcts[1:])))
    check("progress reaches the total", pcts[-1][0] == pcts[-1][1])
    check("progress totals compressed bytes",
          pcts[-1][1] == sum(f["xz_size"] for f in small))

    # 2. Republishing over a live payload must not destroy it.
    cuda_download.download_payload(None)
    check("republishing keeps a working payload", cuda_setup.payload_present(root))
    check("no cuda-old is left behind", not (root.parent / "cuda-old").exists())

    # 3. A failure part-way must leave the finished asset for the retry to reuse.
    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(root.parent / "cuda-staging", ignore_errors=True)
    seen.clear()
    _httpx.Client = lambda **kw: FakeClient(blobs, fail_on="b.dll.xz", record=seen)
    failed = False
    try:
        cuda_download.download_payload(None)
    except OSError:
        failed = True
    check("a failed download raises", failed)
    check("a failed download publishes nothing", not cuda_setup.payload_present(root))
    check("staging survives a failure so a retry can resume",
          (root.parent / "cuda-staging").exists(),
          "clearing it on failure is what made the Range resume dead code")

    seen.clear()
    _httpx.Client = lambda **kw: FakeClient(blobs, record=seen)
    cuda_download.download_payload(None)
    check("the retry succeeds", cuda_setup.payload_present(root))
    check("the retry does not refetch the completed asset",
          "a.dll.xz" not in [n for n, _ in seen],
          f"refetched: {seen}")

    # 4. A corrupt asset must never be published.
    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(root.parent / "cuda-staging", ignore_errors=True)
    bad = dict(blobs)
    bad["a.dll.xz"] = lzma.compress(b"\x01" * 2048, preset=1)
    _httpx.Client = lambda **kw: FakeClient(bad)
    rejected = False
    try:
        cuda_download.download_payload(None)
    except RuntimeError as exc:
        rejected = "checksum" in str(exc)
    check("a corrupt asset is rejected on its digest", rejected)
    check("a corrupt download publishes nothing", not cuda_setup.payload_present(root))

    _httpx.Client = real_client
    cuda_download.manifest = real_manifest
    cuda_setup.manifest = real_cs_manifest
    cuda_download.download_root = real_dl
    cuda_setup.download_root = real_cs_dl


# ------------------------------------------------------------------ report
print(f"\n{checks} checks")
if failures:
    print(f"\n{len(failures)} FAILED:")
    for f in failures:
        print(f"  - {f}")
    raise SystemExit(1)
print("ALL PASS")
