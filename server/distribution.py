"""Features excluded from the Microsoft Store package."""

from pathlib import Path

# Written only by packaging into the installed backend, never into user preferences.
IS_STORE_BUILD = (Path(__file__).resolve().parent.parent / "store-build").is_file()


def model_allowed(model_id: str | None) -> bool:
    return not (IS_STORE_BUILD and model_id == "windows-speech")


def startup_model(requested: str | None) -> str | None:
    """An old preview selection returns to the ordinary hardware-based model choice."""
    return requested if model_allowed(requested) else None
