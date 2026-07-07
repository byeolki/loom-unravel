from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ImageResult:
    filename: str
    success: bool
    missing_labels: list[str]
    error: str | None
    artifact_review: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationSummary:
    results: list[ImageResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def parts_pass_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def parts_pass_rate(self) -> float:
        return self.parts_pass_count / self.total if self.total else 0.0

    @property
    def meets_parts_criterion(self) -> bool:
        return self.total > 0 and self.parts_pass_rate >= 0.7

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "parts_pass_count": self.parts_pass_count,
            "parts_pass_rate": self.parts_pass_rate,
            "meets_parts_criterion": self.meets_parts_criterion,
            "results": [r.to_dict() for r in self.results],
        }
