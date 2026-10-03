"""Capture selection is an endpoint identity, never a cross-process array offset."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .coreaudio import CaptureError


def _name_key(name):
    # Match the display-name cleanup used by the native picker for resource strings.
    if "@" in name:
        opening = name.rfind("(")
        closing = name.find(")", opening + 1)
        if opening >= 0 and closing > opening + 1:
            inner = name[opening + 1:closing].strip()
            prefix = name[:name.find("(")].strip()
            if inner:
                name = prefix + " (" + inner + ")"
    return "".join(ch.lower() for ch in name if ch.isalnum())


@dataclass(frozen=True)
class AudioTarget:
    kind: str = "microphone"
    endpoint_id: str | None = None
    name: str | None = None
    index: int | str | None = None
    follow_default: bool = False

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get("kind", "microphone") not in ("microphone", "loopback"):
            raise ValueError("Choose a microphone or system audio input.")
        for field in ("endpoint_id", "name"):
            if value.get(field) is not None and (not isinstance(value[field], str) or len(value[field]) > 1024):
                raise ValueError("Invalid input selection.")
        index = value.get("index")
        if index is not None and (isinstance(index, bool) or not isinstance(index, (int, str))):
            raise ValueError("Invalid input selection.")
        if not isinstance(value.get("follow_default", False), bool):
            raise ValueError("Invalid input selection.")
        return cls(**{field: value[field] for field in cls.__dataclass_fields__ if field in value})

    @classmethod
    def from_settings(cls, settings):
        kind = settings.input_kind or ("loopback" if settings.loopback_device is not None else "microphone")
        index = settings.loopback_device if kind == "loopback" else settings.input_device
        return cls(kind, settings.input_endpoint_id, settings.input_device_name, index,
                   settings.follow_default_input or
                   (index is None and not settings.input_endpoint_id and not settings.input_device_name))

    def to_dict(self):
        return asdict(self)

    def resolve(self, devices, legacy_devices=None):
        candidates = [d for d in devices if bool(d.get("loopback")) == (self.kind == "loopback")]
        if self.follow_default:
            flag = "is_default_output" if self.kind == "loopback" else "is_default_input"
            matches = [d for d in candidates if d.get(flag)]
        elif self.endpoint_id:
            matches = [d for d in candidates if d.get("endpoint_id") == self.endpoint_id]
        else:
            # Migrate names before considering an old index. Never capture an unrelated
            # endpoint merely because Windows assigned it the remembered number.
            name = self.name or (self.index if isinstance(self.index, str) else None)
            if name is None and self.index is not None:
                lookup = legacy_devices if legacy_devices is not None else (
                    candidates if not any(d.get("endpoint_id") for d in candidates) else [])
                legacy = next((d for d in lookup if d["index"] == self.index), None)
                name = legacy["name"] if legacy else None
            matches = [d for d in candidates if name and _name_key(d["name"]) == _name_key(name)]
        if len(matches) > 1:
            raise CaptureError("device_ambiguous", "Choose the input again to identify the correct device.", False)
        if not matches:
            raise CaptureError("device_unavailable", "The selected input is not available. Sunno will reconnect when it returns.")
        found = matches[0]
        return replace(self, endpoint_id=found.get("endpoint_id"), name=found["name"], index=found["index"])
