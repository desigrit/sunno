"""Read the first referenced test utterances from Google's pinned FLEURS release (CC BY 4.0)."""
import csv
import io
import json
from pathlib import Path
import tarfile
import urllib.request

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
CONFIGS = {"en": "en_us", "hi": "hi_in", "es": "es_419", "ja": "ja_jp",
           "ar": "ar_eg", "zh": "cmn_hans_cn", "fr": "fr_fr", "de": "de_de",
           "it": "it_it", "pt": "pt_br"}


def fetch(directory, language, count=2):
    directory = Path(directory) / language
    directory.mkdir(parents=True, exist_ok=True)
    index = directory / "clips.json"
    if index.exists():
        clips = json.loads(index.read_text(encoding="utf-8"))
        if len(clips) >= count and all((directory / c["file"]).is_file() for c in clips):
            return [(directory / c["file"], c["text"]) for c in clips[:count]]
    base = f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}/data/{CONFIGS[language]}"
    with urllib.request.urlopen(base + "/test.tsv", timeout=90) as response:
        rows = csv.reader(io.TextIOWrapper(response, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)
        refs = {row[1]: row[2] for row in rows if len(row) >= 3}
    clips = []
    # Read only the beginning of each compressed archive. No full dataset download.
    with urllib.request.urlopen(base + "/audio/test.tar.gz", timeout=90) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for entry in archive:
                name = Path(entry.name).name
                if entry.isfile() and name in refs and 0 < entry.size <= 4_000_000:
                    (directory / name).write_bytes(archive.extractfile(entry).read())
                    clips.append({"file": name, "text": refs[name]})
                    if len(clips) == count:
                        break
    if len(clips) != count:
        raise RuntimeError(f"Missing {language} test speech")
    index.write_text(json.dumps(clips, ensure_ascii=False), encoding="utf-8")
    return [(directory / c["file"], c["text"]) for c in clips]

