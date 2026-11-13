"""The serializable recipe boundary; importing it never loads TorchTitan."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping

from .config import ModelConfig, TrainingRecipe


@dataclass(frozen=True)
class RecipeSpec:
    """One selected method path, its common-stack adaptations and run defaults.

    ``training`` describes the objective/depth dynamics. ``run`` contains the
    ordinary launch defaults; a qualification job can explicitly override them.
    Evidence is empty until actual runs supply artifact identities.
    """

    name: str
    model: ModelConfig
    training: TrainingRecipe
    source: dict[str, Any] = field(default_factory=dict)
    preserved: tuple[str, ...] = ()
    adaptations: tuple[str, ...] = ()
    optimizer: dict[str, Any] = field(default_factory=dict)
    inference: dict[str, Any] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)
    evidence: tuple[dict[str, Any], ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported recipe schema version: {self.schema_version}")
        self.training.validate_model(self.model)
        if self.inference.get("readout", "terminal") != "terminal":
            raise ValueError("V1 recipes support terminal readout")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["model"] = self.model.to_dict()
        # JSON arrays are used explicitly, so roundtrips do not depend on the
        # serializer silently converting Python tuples.
        value["training"]["sampled_visits"] = list(self.training.sampled_visits)
        for name in ("preserved", "adaptations", "evidence"):
            value[name] = list(value[name])
        return value

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "RecipeSpec":
        unknown = set(values) - {entry.name for entry in fields(cls)}
        if unknown:
            raise ValueError(f"unknown recipe fields: {sorted(unknown)}")
        value = dict(values)
        value["model"] = ModelConfig.from_dict(value["model"])
        training = dict(value["training"])
        training["sampled_visits"] = tuple(training.get("sampled_visits", ()))
        value["training"] = TrainingRecipe(**training)
        for name in ("preserved", "adaptations", "evidence"):
            if name in value:
                value[name] = tuple(value[name])
        return cls(**value)
