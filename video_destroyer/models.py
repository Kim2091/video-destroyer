from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class PairRecord:
    id: str
    key: str
    origin: str
    ownership: str
    hr_path: str
    lr_path: str
    status: str = "discovered"
    hr: Optional[Dict[str, Any]] = None
    lr: Optional[Dict[str, Any]] = None
    rejection_reason: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass
class SequenceRecord:
    id: str
    pair_id: str
    start_frame: int
    frame_count: int
    hr_files: list[str]
    lr_files: list[str]
    status: str = "extracted"
    rejection_reason: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)
