"""MOTChallenge sequence directory adapter."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MOTChallengeSequenceInfo:
    """Parsed metadata and standard paths for one MOTChallenge sequence."""

    sequence_dir: Path
    name: str
    seqinfo_path: Path
    detection_path: Path
    gt_path: Path | None
    image_dir: Path | None
    seq_length: int | None
    frame_rate: int | None
    image_width: int | None
    image_height: int | None
    image_extension: str | None


def load_motchallenge_sequence_info(
    sequence_dir: str | Path,
    require_detection: bool = True,
    require_gt: bool = False,
) -> MOTChallengeSequenceInfo:
    """Load MOTChallenge sequence metadata from a sequence directory."""
    sequence_path = Path(sequence_dir)
    if not sequence_path.exists():
        raise FileNotFoundError(f"MOTChallenge sequence_dir does not exist: {sequence_path}")
    if not sequence_path.is_dir():
        raise FileNotFoundError(f"MOTChallenge sequence_dir is not a directory: {sequence_path}")

    seqinfo_path = sequence_path / "seqinfo.ini"
    if not seqinfo_path.exists():
        raise FileNotFoundError(f"MOTChallenge seqinfo.ini was not found: {seqinfo_path}")

    detection_path = sequence_path / "det" / "det.txt"
    if require_detection and not detection_path.exists():
        raise FileNotFoundError(f"MOTChallenge det.txt was not found: {detection_path}")

    gt_candidate = sequence_path / "gt" / "gt.txt"
    if require_gt and not gt_candidate.exists():
        raise FileNotFoundError(f"MOTChallenge gt.txt was not found: {gt_candidate}")
    gt_path = gt_candidate if gt_candidate.exists() else None

    parser = configparser.ConfigParser()
    parser.read(seqinfo_path, encoding="utf-8")
    section = parser["Sequence"] if parser.has_section("Sequence") else {}

    name = _get_text(section, "name") or sequence_path.name
    image_subdir = _get_text(section, "imDir")

    return MOTChallengeSequenceInfo(
        sequence_dir=sequence_path,
        name=name,
        seqinfo_path=seqinfo_path,
        detection_path=detection_path,
        gt_path=gt_path,
        image_dir=sequence_path / image_subdir if image_subdir else None,
        seq_length=_get_int(section, "seqLength"),
        frame_rate=_get_int(section, "frameRate"),
        image_width=_get_int(section, "imWidth"),
        image_height=_get_int(section, "imHeight"),
        image_extension=_get_text(section, "imExt"),
    )


def _get_text(section: configparser.SectionProxy | dict[str, str], key: str) -> str | None:
    value = section.get(key) if hasattr(section, "get") else None
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _get_int(section: configparser.SectionProxy | dict[str, str], key: str) -> int | None:
    value = _get_text(section, key)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"MOTChallenge seqinfo field {key} must be an integer: {value}") from exc
