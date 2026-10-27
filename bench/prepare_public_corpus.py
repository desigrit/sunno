"""Prepare a small, fixed public speech sanity check, not an accent benchmark.

Requires pyarrow and soundfile in the benchmark environment only.
Source: https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy
The checksum pins the exact audio and reference text across machines.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

SHA256 = "4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603"


def main():
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    path = Path(hf_hub_download("hf-internal-testing/librispeech_asr_dummy",
                               "clean/validation-00000-of-00001.parquet", repo_type="dataset"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Public corpus changed. Review it before updating the pinned checksum.")
    rows = sorted(pq.read_table(path).to_pylist(), key=lambda row: row["id"])
    args.output.mkdir(parents=True, exist_ok=True)
    entries = []
    for row in rows[:args.count]:
        samples, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if rate != 16000 or samples.ndim != 1:
            raise ValueError("Unexpected corpus audio format")
        name = str(row["id"]) + ".wav"
        sf.write(args.output / name, samples, rate, subtype="PCM_16")
        entries.append({"id": str(row["id"]), "wav": name, "reference": row["text"],
                        "speaker_id": row["speaker_id"], "audio_s": len(samples) / rate})
    (args.output / "corpus.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"Prepared {len(entries)} clips, {sum(e['audio_s'] for e in entries):.1f}s, "
          f"{len({e['speaker_id'] for e in entries})} speaker(s). Clean read speech only.")


if __name__ == "__main__":
    main()
