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

    @classmethod
    def from_dict(cls, data: dict) -> "ImageResult":
        return cls(
            filename=data["filename"],
            success=data["success"],
            missing_labels=data["missing_labels"],
            error=data["error"],
            artifact_review=data.get("artifact_review"),
        )


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

    @property
    def artifact_reviewed_results(self) -> list[ImageResult]:
        return [r for r in self.results if r.artifact_review is not None]

    @property
    def artifact_reviewed_count(self) -> int:
        return len(self.artifact_reviewed_results)

    @property
    def artifact_pass_count(self) -> int:
        return sum(1 for r in self.artifact_reviewed_results if r.artifact_review)

    @property
    def artifact_pass_rate(self) -> float | None:
        reviewed = self.artifact_reviewed_count
        return self.artifact_pass_count / reviewed if reviewed else None

    @property
    def meets_artifact_criterion(self) -> bool | None:
        rate = self.artifact_pass_rate
        return rate >= 0.7 if rate is not None else None

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "parts_pass_count": self.parts_pass_count,
            "parts_pass_rate": self.parts_pass_rate,
            "meets_parts_criterion": self.meets_parts_criterion,
            "artifact_reviewed_count": self.artifact_reviewed_count,
            "artifact_pass_rate": self.artifact_pass_rate,
            "meets_artifact_criterion": self.meets_artifact_criterion,
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ValidationSummary":
        return cls(results=[ImageResult.from_dict(r) for r in data["results"]])
