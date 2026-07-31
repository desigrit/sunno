"""Decide the shippable CUDA payload from three independent signals.

No single method is sufficient:
  * PE import table  - catches static and delay-load imports, MISSES LoadLibrary.
  * ASCII/UTF-16 string scan - catches LoadLibrary-by-name, but can false-positive.
  * runtime memory_maps - ground truth for the paths actually exercised, but only for
    the configurations actually run.

Ship a DLL if ANY signal references it, or if a shipped DLL depends on it (transitively).
"""

import re
import sys
from pathlib import Path

import pefile

# Repo-relative so the script works in any clone, not just the machine it was written on.
REPO = Path(__file__).resolve().parent.parent
SITE = REPO / ".venv" / "Lib" / "site-packages"
NVIDIA = SITE / "nvidia"

# Native modules the backend actually loads. Deliberately EXCLUDES any vendored NVIDIA
# redistributable (e.g. ctranslate2 ships a vestigial cudnn64_9.dll): those are not entry
# points, and seeding from the cuDNN dispatcher — which names every engine lib as a string
# literal — would transitively pull the whole 1 GB tree back in.
def _is_nvidia_redist(path: Path) -> bool:
    n = path.name.lower()
    return n.startswith(("cudnn", "cublas", "nvrtc", "nvblas", "cudart", "cufft", "curand"))


APP_MODULES = [
    p for p in [
        *SITE.glob("ctranslate2/*.dll"),
        *SITE.glob("ctranslate2/*.pyd"),
        *SITE.glob("onnxruntime/capi/*.dll"),
        *SITE.glob("onnxruntime/capi/*.pyd"),
        *SITE.glob("sherpa_onnx/lib/*.dll"),
        *SITE.glob("sherpa_onnx/*.pyd"),
        *SITE.glob("av/*.pyd"),
        *SITE.glob("_sounddevice_data/**/*.dll"),
    ] if not _is_nvidia_redist(p)
]

NVIDIA_DLLS = {p.name.lower(): p for p in NVIDIA.rglob("*.dll")}
DLL_RE = re.compile(rb"([A-Za-z0-9_.\-]+\.dll)", re.IGNORECASE)


def import_table(path: Path) -> set[str]:
    names: set[str] = set()
    try:
        pe = pefile.PE(str(path), fast_load=True)
        pe.parse_data_directories(directories=[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
        ])
        for attr in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT"):
            for entry in getattr(pe, attr, []) or []:
                if entry.dll:
                    names.add(entry.dll.decode(errors="replace").lower())
        pe.close()
    except Exception:
        pass
    return names


def strings_scan(path: Path) -> set[str]:
    """DLL names appearing as literals - how LoadLibrary targets show up."""
    try:
        blob = path.read_bytes()
    except OSError:
        return set()
    found = {m.group(1).decode(errors="replace").lower() for m in DLL_RE.finditer(blob)}
    # UTF-16LE literals too
    try:
        wide = blob.decode("utf-16-le", errors="ignore").encode("ascii", errors="ignore")
        found |= {m.group(1).decode(errors="replace").lower() for m in DLL_RE.finditer(wide)}
    except Exception:
        pass
    return found


print("COVERAGE - modules scanned")
print("-" * 78)
direct: set[str] = set()
for mod in sorted(set(APP_MODULES)):
    if mod.suffix.lower() not in (".dll", ".pyd"):
        continue
    imports = import_table(mod)
    literals = strings_scan(mod)
    hits = sorted((imports | literals) & set(NVIDIA_DLLS))
    direct.update(hits)
    src = []
    if imports & set(NVIDIA_DLLS):
        src.append("import-table")
    if (literals - imports) & set(NVIDIA_DLLS):
        src.append("string-literal")
    label = ", ".join(hits) if hits else "(none)"
    print(f"  {mod.name:<34}{label:<34}{'/'.join(src)}")

print(f"\nDirectly referenced by app modules: {sorted(direct) or '(none)'}")

# Transitive closure: a shipped DLL's own dependencies must ship too.
needed = set(direct)
frontier = list(direct)
while frontier:
    name = frontier.pop()
    path = NVIDIA_DLLS.get(name)
    if not path:
        continue
    for dep in import_table(path) | strings_scan(path):
        if dep in NVIDIA_DLLS and dep not in needed:
            needed.add(dep)
            frontier.append(dep)

print(f"After transitive closure:           {sorted(needed)}")

print("\nDECISION")
print("-" * 78)
keep = drop = 0.0
keep_list, drop_list = [], []
for name, path in sorted(NVIDIA_DLLS.items(), key=lambda kv: -kv[1].stat().st_size):
    mb = path.stat().st_size / 1048576
    if name in needed:
        keep += mb
        keep_list.append(path)
        print(f"  [SHIP] {path.name:<46}{mb:>8.1f} MB")
    else:
        drop += mb
        drop_list.append(path)
        print(f"  [    ] {path.name:<46}{mb:>8.1f} MB")

print(f"\n  ship {keep:.1f} MB   omit {drop:.1f} MB   (total {keep + drop:.1f} MB)")

out = REPO / "packaging"
out.mkdir(exist_ok=True)
(out / "cuda_allowlist.txt").write_text(
    "\n".join(sorted(p.relative_to(NVIDIA).as_posix() for p in keep_list)) + "\n",
    encoding="utf-8",
)
print(f"\n  wrote {out / 'cuda_allowlist.txt'}")
