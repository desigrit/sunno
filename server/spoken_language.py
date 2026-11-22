"""Language settings, model capabilities and speech-only Whisper language detection."""
from __future__ import annotations

from dataclasses import dataclass
import threading

import numpy as np

WHISPER_CODES = (
    "af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi fo fr gl gu ha haw he "
    "hi hr ht hu hy id is it ja jw ka kk km kn ko la lb ln lo lt lv mg mi mk ml mn mr ms mt my ne "
    "nl nn no oc pa pl ps pt ro ru sa sd si sk sl sn so sq sr su sv sw ta te tg th tk tl tr tt "
    "uk ur uz vi yi yo zh"
).split()
MULTILINGUAL = {"tiny", "base", "small", "medium", "large-v3"}


def root_language(code):
    return code.replace("_", "-").split("-")[0].lower()


def whisper_codes(model):
    return tuple(WHISPER_CODES) + (("yue",) if model == "large-v3" else ())


def resolve(preference, choices):
    if preference in choices:
        return preference
    # Apple exposes regional locales. An equivalent locale restores a Whisper preference.
    for desired in (preference, "en"):
        if desired and desired != "auto":
            equivalents = [c for c in choices if root_language(c) == root_language(desired)]
            if equivalents:
                return next((c for c in equivalents if c in ("en-US", "zh-CN", "zh-Hans-CN")),
                            equivalents[0])
    return "auto" if "auto" in choices else choices[0]


@dataclass(frozen=True)
class Choice:
    code: str
    revision: int


class SpokenLanguage:
    def __init__(self, model, preference, emit, locales=None):
        self.model = model
        self.choices = (tuple(locales) if locales is not None else
                        ("auto", *whisper_codes(model)) if model in MULTILINGUAL else ("en",))
        if not self.choices:
            raise ValueError("No speech languages are available for this model.")
        self._lock = threading.RLock()
        self._choice = Choice(resolve(preference or "auto", self.choices), 0)
        self.detected = None
        self.emit = emit

    def choice(self):
        with self._lock:
            return self._choice

    def state(self, **extra):
        with self._lock:
            return dict(type="spoken_language", model=self.model, choices=self.choices,
                        selected=self._choice.code, detected=self.detected, **extra)

    def select(self, code):
        if code not in self.choices:
            raise ValueError("This speech model does not support that language.")
        with self._lock:
            if self._choice.code != code:
                self._choice = Choice(code, self._choice.revision + 1)
                self.detected = None
            self.emit(self.state(accepted=code))

    def identified(self, choice, code):
        with self._lock:
            if choice == self._choice and code != self.detected:
                self.detected = code
                self.emit(self.state())


class PhraseLanguages:
    """Used only by the decoder thread. VAD supplies immutable snapshots of voiced frames."""
    WINDOW = 48000
    INITIAL = 12000
    CONFIDENCE = 0.80

    def __init__(self, session):
        self.session = session
        self.choice = None
        self.utterance = None
        self.cursor = 0
        self.buffer = np.empty(0, np.float32)
        self.established = self.candidate = self.guess = None

    def for_job(self, engine, job):
        choice = job.choice
        if choice != self.choice:
            self.choice = choice
            self.utterance = None
            self.buffer = np.empty(0, np.float32)
            self.established = self.candidate = self.guess = None
        if choice.code != "auto":
            return choice.code
        if self.utterance != job.utterance_id:
            self.utterance = job.utterance_id
            self.cursor = 0
        frames = job.voiced[self.cursor:]
        self.cursor = len(job.voiced)
        if frames:
            self.buffer = np.concatenate((self.buffer, *frames))
        # The first short sample provides provisional captions. Only independent full
        # windows can overturn a confident decision. Leftovers carry across pauses.
        while len(self.buffer) >= self.WINDOW or (
                self.guess is None and len(self.buffer) > 0 and (len(self.buffer) >= self.INITIAL or job.is_final)):
            count = min(len(self.buffer), self.WINDOW)
            window, self.buffer = self.buffer[:count], self.buffer[count:]
            code, confidence = engine.detect_language(window)
            if code not in self.session.choices or code == "auto":
                continue
            if self.established is None:
                self.guess = code
                if confidence >= self.CONFIDENCE:
                    self.established = code
                    self.session.identified(choice, code)
            elif confidence < self.CONFIDENCE or code == self.established:
                self.candidate = None
            elif self.candidate == code:
                self.established, self.candidate = code, None
                self.session.identified(choice, code)
            else:
                self.candidate = code
        return self.established or self.guess


def pin(engine, language):
    setter = getattr(engine, "set_language", None)
    if setter is not None:
        setter(language)

