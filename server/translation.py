"""Optional local M2M100 translation. Live ASR never waits for downloads or decoding."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import threading
import json

import numpy as np

from .spoken_language import root_language

REPO = "Xenova/m2m100_418M"
REVISION = "9c374f0b7aca709787cea97b047bfbbd1559d177"
FILES = ("sentencepiece.bpe.model", "vocab.json", "onnx/encoder_model_quantized.onnx",
         "onnx/decoder_model_merged_quantized.onnx")
LANGUAGES = tuple(("af am ar ast az ba be bg bn br bs ca ceb cs cy da de el en es et fa ff fi fr fy "
                   "ga gd gl gu ha he hi hr ht hu hy id ig ilo is it ja jv ka kk km kn ko lb lg ln lo "
                   "lt lv mg mk ml mn mr ms my ne nl no ns oc or pa pl ps pt ro ru sd si sk sl so sq "
                   "sr ss su sv sw ta th tl tn tr uk ur uz vi wo xh yi yo zh zu").split())


def source_code(code):
    root = root_language(code or "")
    return "jv" if root == "jw" else root


class LocalTranslator:
    def __init__(self, directory=None):
        import onnxruntime as ort
        from sentencepiece import SentencePieceProcessor

        if directory is None:
            from huggingface_hub import snapshot_download
            try:
                directory = snapshot_download(REPO, revision=REVISION, allow_patterns=FILES, local_files_only=True)
                if not all((Path(directory) / f).is_file() for f in FILES):
                    directory = None
            except FileNotFoundError:
                pass
            if directory is None:
                directory = snapshot_download(REPO, revision=REVISION, allow_patterns=FILES)
        root = Path(directory)
        self.tokenizer = SentencePieceProcessor(model_file=str(root / FILES[0]))
        self.vocabulary = json.loads((root / FILES[1]).read_text(encoding="utf-8"))
        self.pieces = {identifier: piece for piece, identifier in self.vocabulary.items()}
        self.language_ids = {code: len(self.vocabulary) + i for i, code in enumerate(LANGUAGES)}
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.encoder, self.decoder = (ort.InferenceSession(str(root / f), options,
                                         providers=["CPUExecutionProvider"]) for f in FILES[2:])
        self.cache_names = [i.name for i in self.decoder.get_inputs() if i.name.startswith("past_key_values.")]
        self.outputs = [o.name for o in self.decoder.get_outputs()]

    def translate(self, text, source, target, current=lambda: True):
        src = self.language_ids.get(source)
        dst = self.language_ids.get(target)
        if src is None or dst is None:
            raise ValueError("This language pair is unavailable for translation.")
        pieces = [self.vocabulary.get(piece, 3) for piece in self.tokenizer.encode(text, out_type=str)]
        parts = []
        for start in range(0, len(pieces), 384):
            inputs = np.asarray([[src, *pieces[start:start + 384], 2]], dtype=np.int64)
            mask = np.ones_like(inputs)
            hidden, = self.encoder.run(None, {"input_ids": inputs, "attention_mask": mask})
            # The first pass builds the cross-attention cache and consumes the forced target
            # token. Subsequent passes reuse it and submit only the newly generated token.
            feed = {name: np.empty((1, 16, 0, 64), np.float32) for name in self.cache_names}
            feed.update(encoder_attention_mask=mask, encoder_hidden_states=hidden,
                        use_cache_branch=np.array([False]), input_ids=np.array([[2, dst]], np.int64))
            result = []
            for _ in range(min(512, 3 * inputs.shape[1] + 24)):
                if not current():
                    return ""
                values = self.decoder.run(None, feed)
                token = int(np.argmax(values[0][0, -1]))
                if token == 2:
                    break
                result.append(token)
                for name, value in zip(self.outputs[1:], values[1:]):
                    if value.size:
                        feed[name.replace("present.", "past_key_values.", 1)] = value
                feed.update(use_cache_branch=np.array([True]), input_ids=np.array([[token]], np.int64))
            parts.append(self.tokenizer.decode([self.pieces[i] for i in result if i in self.pieces and i > 3]).strip())
        return ("" if target in {"zh", "ja", "th"} else " ").join(parts)


class TranslationQueue:
    def __init__(self, emit, factory=LocalTranslator):
        self.emit = emit
        self.factory = factory
        self.lock = threading.Condition()
        self.target = "off"
        self.phase = "ready"
        self.revision = 0
        self.pending = OrderedDict()
        self.active = None
        self.thread = None
        self.closed = False
        self.unsupported = None

    def state(self, **extra):
        with self.lock:
            return dict(type="translation_state", selected=self.target, phase=self.phase,
                        choices=("off", *LANGUAGES), **extra)

    def select(self, target, explicit=True):
        if target != "off" and target not in LANGUAGES:
            raise ValueError("This translation language is unavailable.")
        with self.lock:
            self.revision += 1
            self.target = target
            self.pending.clear()
            self.active = self.unsupported = None
            self.phase = "ready" if target == "off" else "preparing"
            self.emit(self.state(**({"accepted": "off"} if explicit and target == "off" else {})))
            if target != "off" and self.thread is None:
                self.thread = threading.Thread(target=self._run, name="translation", daemon=True)
                self.thread.start()
            self.lock.notify_all()

    def submit(self, event):
        source = source_code(event.get("language"))
        with self.lock:
            if self.target == "off" or not event.get("text") or self.closed:
                return
            key = event["id"], event["started_at"]
            if self.active == key:
                self.active = None
            self.pending.pop(key, None)
            if source == self.target:
                return
            if source not in LANGUAGES:
                if self.unsupported != source:
                    self.unsupported = source
                    self.emit(self.state(message="Translation is unavailable for this spoken language."))
                return
            if len(self.pending) >= 64:
                self.pending.popitem(last=False)
                self.emit(self.state(message="Translation fell behind. Older captions keep their original text."))
            self.pending[key] = (event["text"], source)
            self.lock.notify_all()

    def _run(self):
        model = None
        while True:
            with self.lock:
                if self.closed:
                    return
                if self.target == "off":
                    model = None
                if self.target == "off" or (self.phase == "ready" and not self.pending):
                    self.lock.wait()
                    continue
                target, revision = self.target, self.revision
            try:
                if model is None:
                    model = self.factory()
                with self.lock:
                    if revision != self.revision or self.closed:
                        continue
                    if self.phase != "ready":
                        self.phase = "ready"
                        self.emit(self.state(accepted=target))
                    if not self.pending:
                        continue
                    key, (original, source) = self.pending.popitem(last=False)
                    self.active = key

                def current():
                    with self.lock:
                        return not self.closed and revision == self.revision and self.active == key

                translated = model.translate(original, source, target, current)
                with self.lock:
                    if translated and current():
                        self.emit(dict(type="translation", id=key[0], started_at=key[1], original=original,
                                       text=translated, language=source, target_language=target))
                    self.active = None
            except Exception as exc:
                with self.lock:
                    if revision != self.revision:
                        continue
                    self.target, self.phase = "off", "ready"
                    self.pending.clear()
                    self.active = None
                    self.emit(self.state(message=f"Translation stopped. Select a language to retry. {exc}"))

    def close(self):
        with self.lock:
            self.closed = True
            self.pending.clear()
            self.lock.notify_all()

