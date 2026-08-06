"""Fail when a staged payload contains PE binaries for the wrong architecture."""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path


MACHINE_NAMES = {
    0x014C: "x86",
    0x8664: "x64",
    0xAA64: "arm64",
}
EXPECTED_MACHINES = {
    "x64": 0x8664,
    "arm64": 0xAA64,
}
COMIMAGE_FLAGS_ILONLY = 0x00000001
COMIMAGE_FLAGS_32BITREQUIRED = 0x00000002
IMAGE_SCN_MEM_EXECUTE = 0x20000000


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _rva_to_offset(data: bytes, section_table: int, section_count: int, rva: int) -> int:
    for index in range(section_count):
        section = section_table + index * 40
        virtual_size = _u32(data, section + 8)
        virtual_address = _u32(data, section + 12)
        raw_size = _u32(data, section + 16)
        raw_offset = _u32(data, section + 20)
        if virtual_address <= rva < virtual_address + max(virtual_size, raw_size):
            return raw_offset + rva - virtual_address
    raise ValueError(f"RVA 0x{rva:x} is outside every PE section")


def pe_architecture(path: Path) -> str | None:
    data = path.read_bytes()
    if len(data) < 64 or data[:2] != b"MZ":
        return None

    pe_offset = _u32(data, 0x3C)
    if data[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise ValueError("invalid PE signature")

    coff = pe_offset + 4
    machine = _u16(data, coff)
    section_count = _u16(data, coff + 2)
    optional_size = _u16(data, coff + 16)
    optional = coff + 20
    magic = _u16(data, optional)
    data_directories = optional + (112 if magic == 0x20B else 96 if magic == 0x10B else 0)
    if not data_directories:
        raise ValueError(f"unknown optional-header magic 0x{magic:x}")

    section_table = optional + optional_size
    executable_section = any(
        _u32(data, section_table + index * 40 + 36) & IMAGE_SCN_MEM_EXECUTE
        for index in range(section_count)
    )
    # Resource-only DLLs can retain any producer machine tag despite containing no code.
    if not executable_section:
        return "neutral"

    # AnyCPU .NET assemblies use the x86 COFF machine while carrying only IL.
    cli_directory = data_directories + 14 * 8
    cli_rva = _u32(data, cli_directory)
    if machine == 0x014C and cli_rva:
        cli_offset = _rva_to_offset(data, section_table, section_count, cli_rva)
        flags = _u32(data, cli_offset + 16)
        if flags & COMIMAGE_FLAGS_ILONLY and not flags & COMIMAGE_FLAGS_32BITREQUIRED:
            return "anycpu"

    return MACHINE_NAMES.get(machine, f"machine-0x{machine:04x}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected", choices=EXPECTED_MACHINES, required=True)
    parser.add_argument(
        "--allow-cross-arch-file",
        action="append",
        nargs=3,
        metavar=("RELATIVE_PATH", "ARCHITECTURE", "SHA256"),
        default=[],
        help="Pinned relative path, architecture, and hash for an intentional foreign PE",
    )
    args = parser.parse_args()

    expected = args.expected
    failures: list[tuple[Path, str]] = []
    checked = 0
    for path in args.root.rglob("*"):
        if not path.is_file():
            continue
        try:
            architecture = pe_architecture(path)
        except (OSError, struct.error, ValueError) as exc:
            failures.append((path, f"invalid PE: {exc}"))
            continue
        if architecture is None:
            continue
        checked += 1
        if architecture in {expected, "anycpu", "neutral"}:
            continue

        relative = path.relative_to(args.root).as_posix().casefold()
        allowed = any(
            relative == allowed_path.replace("\\", "/").casefold()
            and architecture == allowed_architecture.casefold()
            and hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash.casefold()
            for allowed_path, allowed_architecture, expected_hash in args.allow_cross_arch_file
        )
        if not allowed:
            failures.append((path, architecture))

    if failures:
        for path, architecture in failures:
            print(
                f"{path}: expected {expected} or architecture-neutral, "
                f"found {architecture}"
            )
        return 1

    print(f"Verified {checked} PE files as {expected} or architecture-neutral")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
