"""Shared, data-driven knowledge base for VPTeA prompt pairs.

The module deliberately has no dataset-size constants.  Categories are inferred from
sample paths (UCF-Crime) or XD-Violence ``_label_`` suffixes and prompt pairs are
created for the labels that are actually present.

Optimizer update JSON uses this canonical strict protocol::

    {"action": "keep"}
    {"action": "update", "updates": [
        {"label": "exact label", "side": "positive|negative",
         "description": "new prompt"}
    ]}

The optimizer receives complete positive/negative pairs for scoped labels as
comparison context, while the update scope remains the only write authority.
The parser also accepts the legacy ``target`` plus ``updates`` mapping format.
Unknown keys, labels, and unscoped label/side changes are rejected before any
state is changed.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Optional


DEFAULT_POSITIVE_TEMPLATE = "There is visible {label}-related abnormal behavior in the video."
DEFAULT_NEGATIVE_TEMPLATE = "The video shows ordinary activity without {label}-related abnormal behavior."
_DATASET_UCF = "ucf"
_DATASET_XD = "xd"
_DATASETS = {_DATASET_UCF, _DATASET_XD}
_TARGETS = {"positive", "negative", "both"}
_ACTIONS = {"keep", "update"}


class KnowledgeBaseError(ValueError):
    """Base error for invalid knowledge-base data or optimizer updates."""


class OptimizerProtocolError(KnowledgeBaseError):
    """Raised when an optimizer response does not follow the JSON protocol."""


@dataclass(frozen=True)
class PromptPair(Mapping[str, str]):
    """The positive and negative prompt for one discovered category."""

    positive: str
    negative: str

    def __getitem__(self, key: str) -> str:
        if key == "positive":
            return self.positive
        if key == "negative":
            return self.negative
        raise KeyError(key)

    def __iter__(self):
        return iter(("positive", "negative"))

    def __len__(self) -> int:
        return 2

    def as_dict(self) -> dict[str, str]:
        return {"positive": self.positive, "negative": self.negative}


@dataclass(frozen=True)
class LearnerDecision:
    """Validated learner prediction and category attribution."""

    prediction: int
    matched_labels: tuple[str, ...] = ()
    related_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class OptimizerChange:
    """One applied change to one side of one prompt pair."""

    label: str
    side: str
    description: str

    def as_dict(self) -> dict[str, str]:
        return {
            "label": self.label,
            "side": self.side,
            "description": self.description,
        }


@dataclass(frozen=True)
class OptimizerUpdate:
    """Validated optimizer operation returned by :func:`parse_optimizer_update`."""

    action: str
    target: Optional[str] = None
    updates: Mapping[str, Any] = field(default_factory=dict)
    changes: tuple[OptimizerChange, ...] = ()



def _path_value(item: Any) -> str:
    """Get a path/name from an annotation item or path-like value."""
    if isinstance(item, Mapping):
        for key in ("path", "video", "video_name", "name"):
            value = item.get(key)
            if isinstance(value, (str, os.PathLike)):
                return os.fspath(value)
        raise KnowledgeBaseError(
            "dataset item must contain one of: path, video, video_name, name"
        )
    if isinstance(item, (str, os.PathLike)):
        return os.fspath(item)
    raise KnowledgeBaseError(f"dataset item is not path-like: {item!r}")


def _posix_path(value: str) -> PurePosixPath:
    # PurePosixPath keeps this parser consistent when annotations were generated
    # on Linux but are consumed on Windows (or vice versa).
    return PurePosixPath(value.replace("\\", "/"))


def _is_normal_category(category: str) -> bool:
    """Return whether a UCF parent directory denotes normal video."""
    words = re.findall(r"[a-z0-9]+", category.casefold())
    return "normal" in words


def extract_ucf_category(path_or_item: Any) -> Optional[str]:
    """Extract a UCF category from the immediate parent directory.

    Normal videos (for example ``Normal_Videos``) return ``None``. The check is
    token-based so a real category such as ``Abnormal`` is not accidentally
    discarded.
    """
    path = _posix_path(_path_value(path_or_item))
    category = path.parent.name.strip()
    if not category or _is_normal_category(category):
        return None
    return category


def discover_ucf_categories(paths: Iterable[Any]) -> set[str]:
    """Discover non-normal UCF categories from sample paths or annotations."""
    return {
        category
        for item in paths
        for category in (extract_ucf_category(item),)
        if category is not None
    }


def _xd_label_tokens(path_or_item: Any) -> set[str]:
    path = _posix_path(_path_value(path_or_item))
    name = path.name
    # A label suffix is part of the filename/name, not a directory component.
    match = re.search(r"_label_(.+)$", name, flags=re.IGNORECASE)
    if match is None:
        return set()
    suffix = match.group(1)
    # Remove a filename extension without treating dots in the label as tokens.
    suffix = suffix.rsplit(".", 1)[0]
    tokens = re.split(r"[-_,;+\s]+", suffix)
    return {
        token
        for token in tokens
        if token and token.casefold() not in {"a", "0"}
    }


def extract_xd_categories(path_or_item: Any) -> set[str]:
    """Extract atomic XD-Violence labels, excluding the ``A``/``0`` markers."""
    return _xd_label_tokens(path_or_item)


def discover_xd_categories(paths: Iterable[Any]) -> set[str]:
    """Discover XD-Violence categories from names or annotation items."""
    categories: set[str] = set()
    for item in paths:
        categories.update(_xd_label_tokens(item))
    return categories


# Label-oriented aliases make the discovery functions convenient for callers
# that use the dataset's terminology rather than the knowledge-base terminology.
discover_ucf_labels = discover_ucf_categories
discover_xd_labels = discover_xd_categories


def discover_categories(
    *, ucf_paths: Iterable[Any] = (), xd_paths: Iterable[Any] = ()
) -> dict[str, set[str]]:
    """Discover categories for both datasets without hard-coded class counts."""
    return {
        _DATASET_UCF: discover_ucf_categories(ucf_paths),
        _DATASET_XD: discover_xd_categories(xd_paths),
    }


def _validate_description(text: Any, label: str, side: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise KnowledgeBaseError(f"{side} prompt for {label!r} must be a non-empty string")
    text = text.strip()
    if "?" in text or "\n" in text or "\r" in text:
        raise KnowledgeBaseError(
            f"{side} prompt for {label!r} must be one declarative description"
        )
    if re.match(r"^(?:[-*]|\d+[.)])\s+", text):
        raise KnowledgeBaseError(
            f"{side} prompt for {label!r} must not be a list item"
        )
    return text


def _coerce_pair(value: Any, label: str) -> PromptPair:
    if isinstance(value, PromptPair):
        pair = value
    elif isinstance(value, Mapping):
        if set(value) != _TARGETS - {"both"}:
            raise KnowledgeBaseError(
                f"prompt pair for {label!r} must have positive and negative keys"
            )
        pair = PromptPair(value["positive"], value["negative"])
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        pair = PromptPair(value[0], value[1])
    else:
        raise KnowledgeBaseError(f"invalid prompt pair for {label!r}")
    positive = _validate_description(pair.positive, label, "positive")
    negative = _validate_description(pair.negative, label, "negative")
    if positive == negative:
        raise KnowledgeBaseError(
            f"positive and negative prompts must differ for {label!r}"
        )
    return PromptPair(positive, negative)


def load_semantic_seeds(
    path: str | os.PathLike[str], dataset: str
) -> dict[str, PromptPair]:
    """Load one positive/negative seed pair per raw annotation label."""
    source = Path(path)
    try:
        with source.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeBaseError(f"cannot load semantic seed file: {source}") from exc
    if not isinstance(data, Mapping) or not isinstance(data.get("datasets"), Mapping):
        raise KnowledgeBaseError("semantic seed JSON must contain a datasets object")
    dataset = dataset.casefold()
    pairs = data["datasets"].get(dataset)
    if not isinstance(pairs, Mapping):
        raise KnowledgeBaseError(f"semantic seeds missing dataset {dataset!r}")
    return {label: _coerce_pair(value, label) for label, value in pairs.items()}


def initialize_prompt_pairs(
    labels: Iterable[str],
    *,
    positive_template: str = DEFAULT_POSITIVE_TEMPLATE,
    negative_template: str = DEFAULT_NEGATIVE_TEMPLATE,
    existing: Optional[Mapping[str, Any]] = None,
    seeds: Optional[Mapping[str, Any]] = None,
    require_seeds: bool = False,
) -> dict[str, PromptPair]:
    """Create one prompt pair per label, preserving already learned pairs.

    Labels are normalized to unique, non-empty strings and sorted for stable
    rendering and JSON output. Existing pairs are retained verbatim; newly
    discovered labels use the two supplied templates.
    """
    if not isinstance(positive_template, str) or not positive_template.strip():
        raise KnowledgeBaseError("positive_template must be a non-empty string")
    if not isinstance(negative_template, str) or not negative_template.strip():
        raise KnowledgeBaseError("negative_template must be a non-empty string")
    normalized = set()
    for label in labels:
        if not isinstance(label, str) or not label.strip():
            raise KnowledgeBaseError("labels must be non-empty strings")
        normalized.add(label.strip())
    existing = existing or {}
    seeds = seeds or {}
    if not isinstance(existing, Mapping) or not isinstance(seeds, Mapping):
        raise KnowledgeBaseError("existing prompt pairs and seeds must be mappings")
    pairs: dict[str, PromptPair] = {}
    for label in sorted(normalized):
        if label in existing:
            try:
                current_pair = _coerce_pair(existing[label], label)
                generated_pair = PromptPair(
                    positive_template.format(label=label).strip(),
                    negative_template.format(label=label).strip(),
                )
                if require_seeds and current_pair == generated_pair and label in seeds:
                    pairs[label] = _coerce_pair(seeds[label], label)
                else:
                    pairs[label] = current_pair
                continue
            except KnowledgeBaseError:
                if require_seeds and label not in seeds:
                    raise
        if label in seeds:
            pairs[label] = _coerce_pair(seeds[label], label)
            continue
        if require_seeds:
            raise KnowledgeBaseError(f"missing semantic seed for discovered label {label!r}")
        try:
            positive = positive_template.format(label=label)
            negative = negative_template.format(label=label)
        except (KeyError, IndexError, ValueError) as exc:
            raise KnowledgeBaseError("prompt templates must format with {label}") from exc
        pairs[label] = _coerce_pair(
            {"positive": positive.strip(), "negative": negative.strip()}, label
        )
    return pairs


def render_all(prompt_pairs: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Render all prompt pairs to a JSON-friendly, deterministic mapping."""
    if not isinstance(prompt_pairs, Mapping):
        raise KnowledgeBaseError("prompt_pairs must be a mapping")
    return {
        label: _coerce_pair(prompt_pairs[label], label).as_dict()
        for label in sorted(prompt_pairs)
    }


def render_optimizer_prompt(
    template: str,
    data_token: str,
    samples: Iterable[Mapping[str, Any]],
    update_scope: Mapping[str, Any],
    knowledge_base: "KnowledgeBase",
) -> str:
    """Render optimizer evidence with complete pairs for scoped labels."""
    samples = list(samples)
    rendered = knowledge_base.render_all()
    scoped_pairs = {
        label: rendered[label]
        for label in update_scope
    }
    return (
        template.replace("[$Data]", data_token)
        .replace("[$Samples]", json.dumps(samples, ensure_ascii=False, indent=2))
        .replace("[$KnowledgeBase]", json.dumps(scoped_pairs, ensure_ascii=False, indent=2))
        .replace("[$UpdateScope]", json.dumps(update_scope, ensure_ascii=False, indent=2))
        .replace("[$Prediction]", json.dumps([sample["prediction"] for sample in samples]))
        .replace("[$GroundTruth]", json.dumps([sample["target"] for sample in samples]))
        .replace(
            "[$SampleLabels]",
            json.dumps([sample["true"] for sample in samples], ensure_ascii=False),
        )
        .replace("[$AllowedLabels]", json.dumps(sorted(update_scope), ensure_ascii=False))
    )


def _parse_json_object(
    value: str,
    *,
    response_name: str,
    allow_wrapped: bool = False,
    repair_one_trailing_brace: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeBaseError(f"{response_name} must be a JSON object")
    candidate = value.strip()
    if allow_wrapped:
        full_fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.I | re.S)
        if full_fence:
            candidate = full_fence.group(1).strip()
        else:
            fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.I | re.S)
            if fenced:
                candidate = fenced.group(1)
            else:
                start, end = candidate.find("{"), candidate.rfind("}")
                if start >= 0 and end > start:
                    candidate = candidate[start:end + 1]
        if (
            repair_one_trailing_brace
            and candidate.startswith("{")
            and candidate.endswith("]")
            and candidate.count("{") == candidate.count("}") + 1
        ):
            candidate += "}"

    def reject_duplicates(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise KnowledgeBaseError(f"duplicate JSON key: {key!r}")
            result[key] = item
        return result

    try:
        parsed = json.loads(
            candidate,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                KnowledgeBaseError(f"invalid JSON constant: {token}")
            ),
        )
    except KnowledgeBaseError:
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        raise KnowledgeBaseError(f"{response_name} must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise KnowledgeBaseError(f"{response_name} must be a JSON object")
    return parsed


def _allowed_label_set(allowed_labels: Optional[Iterable[str]]) -> Optional[set[str]]:
    if allowed_labels is None:
        return None
    result = set()
    for label in allowed_labels:
        if not isinstance(label, str) or not label.strip():
            raise OptimizerProtocolError("allowed labels must be non-empty strings")
        result.add(label)
    return result


def _scope_sides(scope: Optional[Mapping[str, Any]]) -> Optional[dict[str, set[str]]]:
    if scope is None:
        return None
    if not isinstance(scope, Mapping):
        raise OptimizerProtocolError("scope must be a mapping")
    result: dict[str, set[str]] = {}
    for label, sides in scope.items():
        if not isinstance(label, str) or not label.strip():
            raise OptimizerProtocolError("scope labels must be non-empty strings")
        if not isinstance(sides, Mapping):
            raise OptimizerProtocolError("scope values must map sides to reasons")
        result[label] = set()
        for side in sides:
            if side not in {"positive", "negative"}:
                raise OptimizerProtocolError("scope sides must be positive or negative")
            result[label].add(side)
    return result


def _check_scope(scope: Optional[dict[str, set[str]]], label: str, side: str) -> None:
    if scope is not None and side not in scope.get(label, set()):
        raise OptimizerProtocolError(
            f"label/side is outside the authorized scope: {label!r}/{side!r}"
        )


def parse_learner_decision(
    response: str | Mapping[str, Any],
    *,
    allowed_labels: Optional[Iterable[str]] = None,
) -> LearnerDecision:
    """Parse the learner's strict prediction and label attribution JSON."""
    data = (
        _parse_json_object(
            response, response_name="learner response", allow_wrapped=True
        )
        if isinstance(response, str)
        else response
    )
    if not isinstance(data, Mapping) or set(data) != {
        "prediction", "matched_labels", "related_labels"
    }:
        raise KnowledgeBaseError(
            "learner response requires prediction, matched_labels, and related_labels"
        )
    prediction = data["prediction"]
    if isinstance(prediction, bool) or prediction not in (0, 1):
        raise KnowledgeBaseError("learner prediction must be integer 0 or 1")
    allowed = _allowed_label_set(allowed_labels)
    parsed_labels = {}
    for key in ("matched_labels", "related_labels"):
        values = data[key]
        if not isinstance(values, list) or any(
            not isinstance(label, str) or not label.strip() for label in values
        ):
            raise KnowledgeBaseError(f"learner {key} must be a list of labels")
        if len(values) != len(set(values)):
            raise KnowledgeBaseError(f"learner {key} must not contain duplicate labels")
        if allowed is not None:
            unknown = set(values) - allowed
            if unknown:
                raise KnowledgeBaseError(f"unknown learner label: {sorted(unknown)!r}")
        parsed_labels[key] = tuple(values)
    if set(parsed_labels["matched_labels"]) & set(parsed_labels["related_labels"]):
        raise KnowledgeBaseError("learner matched and related labels must be disjoint")
    if prediction == 1 and (
        not parsed_labels["matched_labels"] or parsed_labels["related_labels"]
    ):
        raise KnowledgeBaseError(
            "prediction 1 requires matched labels and no related labels"
        )
    if prediction == 0 and parsed_labels["matched_labels"]:
        raise KnowledgeBaseError("prediction 0 cannot contain matched labels")
    return LearnerDecision(
        prediction=prediction,
        matched_labels=parsed_labels["matched_labels"],
        related_labels=parsed_labels["related_labels"],
    )


def parse_optimizer_update(
    response: str | Mapping[str, Any],
    *,
    allowed_labels: Optional[Iterable[str]] = None,
    scope: Optional[Mapping[str, Any]] = None,
) -> OptimizerUpdate:
    """Strictly parse either the canonical or legacy optimizer JSON protocol."""
    try:
        data = (
            _parse_json_object(
                response,
                response_name="optimizer response",
                allow_wrapped=True,
                repair_one_trailing_brace=True,
            )
            if isinstance(response, str)
            else response
        )
    except KnowledgeBaseError as exc:
        raise OptimizerProtocolError(str(exc)) from exc
    if not isinstance(data, Mapping):
        raise OptimizerProtocolError("optimizer response must be a JSON object")
    allowed = _allowed_label_set(allowed_labels)
    checked_scope = _scope_sides(scope)
    action = data.get("action")
    if not isinstance(action, str) or action not in _ACTIONS:
        raise OptimizerProtocolError("action must be exactly 'keep' or 'update'")

    if action == "keep":
        if set(data) != {"action"}:
            raise OptimizerProtocolError("keep accepts only the action key")
        return OptimizerUpdate("keep")

    # Canonical wire format used by VPTeA_optimizer_instruct.txt.
    if "target" not in data and isinstance(data.get("updates"), list):
        if set(data) != {"action", "updates"} or not data["updates"]:
            raise OptimizerProtocolError("update requires a non-empty updates list")
        changes = []
        seen = set()
        for item in data["updates"]:
            if not isinstance(item, Mapping) or set(item) != {"label", "side", "description"}:
                raise OptimizerProtocolError(
                    "each optimizer update requires label, side, and description"
                )
            label, side = item["label"], item["side"]
            if not isinstance(label, str) or not label.strip():
                raise OptimizerProtocolError("updated labels must be non-empty strings")
            if side not in {"positive", "negative"}:
                raise OptimizerProtocolError("update side must be positive or negative")
            if allowed is not None and label not in allowed:
                raise OptimizerProtocolError(f"unknown label: {label!r}")
            _check_scope(checked_scope, label, side)
            key = (label, side)
            if key in seen:
                raise OptimizerProtocolError("duplicate label/side update")
            seen.add(key)
            try:
                description = _validate_description(item["description"], label, side)
            except KnowledgeBaseError as exc:
                raise OptimizerProtocolError(str(exc)) from exc
            changes.append(OptimizerChange(label, side, description))
        return OptimizerUpdate("update", changes=tuple(changes))

    # Legacy protocol retained for callers/tests that still send target+mapping.
    if set(data) != {"action", "target", "updates"}:
        raise OptimizerProtocolError(
            "update requires action and updates, or action, target, and updates"
        )
    target = data["target"]
    updates = data["updates"]
    if not isinstance(target, str) or target not in _TARGETS:
        raise OptimizerProtocolError("target must be positive, negative, or both")
    if not isinstance(updates, Mapping) or not updates:
        raise OptimizerProtocolError("updates must be a non-empty JSON object")

    checked: dict[str, Any] = {}
    for label, value in updates.items():
        if not isinstance(label, str) or not label.strip():
            raise OptimizerProtocolError("updated labels must be non-empty strings")
        if allowed is not None and label not in allowed:
            raise OptimizerProtocolError(f"unknown label: {label!r}")
        if target == "both":
            if not isinstance(value, Mapping) or set(value) != {"positive", "negative"}:
                raise OptimizerProtocolError(
                    "both updates require positive and negative prompt keys"
                )
            pair_update = {}
            for side in ("positive", "negative"):
                _check_scope(checked_scope, label, side)
                try:
                    pair_update[side] = _validate_description(value[side], label, side)
                except KnowledgeBaseError as exc:
                    raise OptimizerProtocolError(str(exc)) from exc
            if pair_update["positive"] == pair_update["negative"]:
                raise OptimizerProtocolError("positive and negative prompts must differ")
            checked[label] = pair_update
        else:
            _check_scope(checked_scope, label, target)
            try:
                checked[label] = _validate_description(value, label, target)
            except KnowledgeBaseError as exc:
                raise OptimizerProtocolError(str(exc)) from exc
    return OptimizerUpdate("update", target, checked)


# Explicit aliases for callers that name the wire format rather than the action.
parse_optimizer_json = parse_optimizer_update


def atomic_save_json(path: str | os.PathLike[str], value: Any) -> None:
    """Write JSON by fsyncing a sibling temporary file before replacing ``path``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


class KnowledgeBase:
    """In-memory shared categories and prompt pairs with optional JSON storage."""

    def __init__(
        self,
        *,
        ucf_paths: Iterable[Any] = (),
        xd_paths: Iterable[Any] = (),
        categories: Optional[Mapping[str, Iterable[str]]] = None,
        prompt_pairs: Optional[Mapping[str, Any]] = None,
        state_path: Optional[str | os.PathLike[str]] = None,
        seeds: Optional[Mapping[str, Any]] = None,
        require_seeds: bool = False,
    ) -> None:
        discovered = discover_categories(ucf_paths=ucf_paths, xd_paths=xd_paths)
        if categories is not None:
            if not isinstance(categories, Mapping):
                raise KnowledgeBaseError("categories must be a mapping")
            for dataset in _DATASETS:
                values = categories.get(dataset, ())
                if isinstance(values, str):
                    values = (values,)
                for label in values:
                    if not isinstance(label, str) or not label.strip():
                        raise KnowledgeBaseError("categories must contain non-empty strings")
                    if dataset == _DATASET_UCF and _is_normal_category(label):
                        continue
                    discovered[dataset].add(label.strip())
        self.categories: dict[str, set[str]] = discovered
        self.prompt_pairs = initialize_prompt_pairs(
            set().union(*self.categories.values()),
            existing=prompt_pairs,
            seeds=seeds,
            require_seeds=require_seeds,
        )
        if seeds:
            repaired = {}
            for label, pair in self.prompt_pairs.items():
                seed_pair = seeds.get(label)
                if seed_pair is not None and pair.positive == pair.negative:
                    repaired[label] = _coerce_pair(seed_pair, label)
                else:
                    repaired[label] = pair
            self.prompt_pairs = repaired
        self.state_path = Path(state_path) if state_path is not None else None

    @property
    def allowed_labels(self) -> frozenset[str]:
        return frozenset().union(*self.categories.values(), self.prompt_pairs.keys())

    @property
    def labels(self) -> frozenset[str]:
        return self.allowed_labels

    def parse_learner_decision(self, response):
        return parse_learner_decision(response, allowed_labels=self.labels)

    def build_update_scope(self, targets, true_label_sets, decisions):
        return build_update_scope(targets, true_label_sets, decisions)

    def add_ucf_paths(self, paths: Iterable[Any], *, seeds=None, require_seeds=False) -> set[str]:
        labels = discover_ucf_categories(paths)
        self.categories[_DATASET_UCF].update(labels)
        self.prompt_pairs = initialize_prompt_pairs(
            self.allowed_labels, existing=self.prompt_pairs,
            seeds=seeds, require_seeds=require_seeds,
        )
        self._repair_collapsed_pairs(seeds)
        return labels

    def add_xd_paths(self, paths: Iterable[Any], *, seeds=None, require_seeds=False) -> set[str]:
        labels = discover_xd_categories(paths)
        self.categories[_DATASET_XD].update(labels)
        self.prompt_pairs = initialize_prompt_pairs(
            self.allowed_labels, existing=self.prompt_pairs,
            seeds=seeds, require_seeds=require_seeds,
        )
        self._repair_collapsed_pairs(seeds)
        return labels

    def _repair_collapsed_pairs(self, seeds=None) -> None:
        if not seeds:
            return
        repaired = dict(self.prompt_pairs)
        for label, pair in repaired.items():
            seed_pair = seeds.get(label)
            if seed_pair is not None and pair.positive == pair.negative:
                repaired[label] = _coerce_pair(seed_pair, label)
        self.prompt_pairs = repaired

    def initialize_prompt_pairs(self, *, seeds=None, require_seeds=False) -> dict[str, PromptPair]:
        self.prompt_pairs = initialize_prompt_pairs(
            self.allowed_labels, existing=self.prompt_pairs,
            seeds=seeds, require_seeds=require_seeds,
        )
        return dict(self.prompt_pairs)

    def render_all(self, dataset: Optional[str] = None) -> dict[str, dict[str, str]]:
        if dataset is None:
            labels = self.allowed_labels
        else:
            dataset = dataset.casefold()
            if dataset not in _DATASETS:
                raise KnowledgeBaseError("dataset must be 'ucf' or 'xd'")
            labels = self.categories[dataset]
        return render_all({label: self.prompt_pairs[label] for label in labels})

    def apply_optimizer_update(
        self,
        response: str | Mapping[str, Any] | OptimizerUpdate,
        *,
        allowed_labels: Optional[Iterable[str]] = None,
        scope: Optional[Mapping[str, Any]] = None,
        save: bool = True,
    ) -> OptimizerUpdate:
        effective_allowed = self.allowed_labels if allowed_labels is None else set(allowed_labels)
        update = (
            response
            if isinstance(response, OptimizerUpdate)
            else parse_optimizer_update(
                response, allowed_labels=effective_allowed, scope=scope
            )
        )
        if update.action == "keep":
            return update
        changes = list(update.changes)
        if not changes:
            for label, value in update.updates.items():
                if update.target == "both":
                    changes.extend((
                        OptimizerChange(label, "positive", value["positive"]),
                        OptimizerChange(label, "negative", value["negative"]),
                    ))
                else:
                    changes.append(OptimizerChange(label, update.target, value))
        unknown = {change.label for change in changes} - set(self.allowed_labels)
        if unknown:
            raise OptimizerProtocolError(f"unknown label: {sorted(unknown)!r}")
        new_pairs = dict(self.prompt_pairs)
        for change in changes:
            current = new_pairs[change.label]
            if change.side == "positive":
                new_pairs[change.label] = PromptPair(change.description, current.negative)
            else:
                new_pairs[change.label] = PromptPair(current.positive, change.description)
            pair = new_pairs[change.label]
            if pair.positive == pair.negative:
                raise OptimizerProtocolError(
                    f"positive and negative prompts must differ for {change.label!r}"
                )
        self.prompt_pairs = new_pairs
        if save and self.state_path is not None:
            self.save()
        return OptimizerUpdate(
            update.action, update.target, update.updates, tuple(changes)
        )

    def update_from_optimizer_json(
        self,
        response: str | Mapping[str, Any],
        *,
        allowed_labels: Optional[Iterable[str]] = None,
        scope: Optional[Mapping[str, Any]] = None,
        save: bool = True,
    ) -> OptimizerUpdate:
        return self.apply_optimizer_update(
            response, allowed_labels=allowed_labels, scope=scope, save=save
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "categories": {
                dataset: sorted(self.categories[dataset]) for dataset in sorted(_DATASETS)
            },
            "prompt_pairs": render_all(self.prompt_pairs),
        }

    def save(self, path: Optional[str | os.PathLike[str]] = None) -> Path:
        target = Path(path) if path is not None else self.state_path
        if target is None:
            raise KnowledgeBaseError("a state_path is required to save the knowledge base")
        atomic_save_json(target, self.to_dict())
        self.state_path = target
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "KnowledgeBase":
        source = Path(path)
        try:
            with source.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise KnowledgeBaseError(f"cannot load knowledge base: {source}") from exc
        if not isinstance(data, Mapping):
            raise KnowledgeBaseError("knowledge-base JSON must be an object")
        version = data.get("version")
        if version not in (1, 2):
            raise KnowledgeBaseError("unsupported knowledge-base schema version")

        categories = data.get("categories", {})
        prompt_pairs = data.get("prompt_pairs", {})
        if not isinstance(categories, Mapping) or not isinstance(prompt_pairs, Mapping):
            raise KnowledgeBaseError("invalid knowledge-base JSON shape")
        return cls(
            categories=categories,
            prompt_pairs=prompt_pairs,
            state_path=source,
        )


def allowed_labels_for_batch(
    targets: Iterable[Any],
    sample_labels: Iterable[Iterable[str]],
    all_labels: Iterable[str],
) -> set[str]:
    """Return labels that the verbalized optimizer may update for one batch."""
    targets = list(targets)
    sample_labels = [set(labels) for labels in sample_labels]
    if len(targets) != len(sample_labels):
        raise KnowledgeBaseError("targets and sample_labels must have the same length")
    allowed: set[str] = set()
    for target, labels in zip(targets, sample_labels):
        # Only false negatives need positive-side repair from their true labels.
        # False positives are scoped later by learner attribution; a correctly
        # classified sample never opens a knowledge-base update scope.
        if int(target) == 1:
            allowed.update(labels)
    return allowed


def build_update_scope(
    targets: Iterable[Any],
    true_label_sets: Iterable[Iterable[str]],
    decisions: Iterable[LearnerDecision],
) -> dict[str, dict[str, list[str]]]:
    """Authorize prompt sides from real labels and prediction errors only."""
    targets = list(targets)
    true_label_sets = [set(labels) for labels in true_label_sets]
    decisions = list(decisions)
    if not (len(targets) == len(true_label_sets) == len(decisions)):
        raise KnowledgeBaseError("targets, true labels, and decisions must have equal lengths")
    scope: dict[str, dict[str, list[str]]] = {}
    for target, true_labels, decision in zip(targets, true_label_sets, decisions):
        if not isinstance(decision, LearnerDecision):
            raise KnowledgeBaseError("decisions must contain LearnerDecision values")
        if int(target) == decision.prediction:
            continue
        if int(target) == 1:
            for label in true_labels:
                sides = scope.setdefault(label, {})
                sides.setdefault("positive", []).append("false_negative_true_label")
                sides.setdefault("negative", []).append("false_negative_true_label")
        else:
            for label in decision.matched_labels:
                scope.setdefault(label, {}).setdefault("negative", []).append("false_positive_matched_label")
    return scope


def _is_xd_annotation(annotation: Iterable[Any]) -> bool:
    """Infer the annotation family from fields and encoded labels."""
    for item in annotation:
        if isinstance(item, Mapping):
            if "video_name" in item or "path" in item:
                return True
            video = item.get("video")
            if isinstance(video, str) and "_label_" in video.casefold():
                return True
    return False


def load_or_create_knowledge_base(
    annotation: Iterable[Any],
    annotation_path: str | os.PathLike[str],
    *,
    state_dir: str | os.PathLike[str] = "Data",
) -> KnowledgeBase:
    """Load or initialize a knowledge base from annotation-derived labels."""
    annotation = list(annotation)
    annotation_path = Path(annotation_path)
    stem = annotation_path.stem.casefold()
    is_xd = _is_xd_annotation(annotation)
    is_ucf = not is_xd and "ucf" in stem
    if not is_xd and not is_ucf:
        raise KnowledgeBaseError(
            "cannot infer dataset family from annotation; include 'ucf' in the "
            "annotation name or use XD-style video_name/path labels"
        )
    state_name = "UCF_knowledge_base.json" if is_ucf else "XD_knowledge_base.json"
    state_path = Path(state_dir) / state_name
    seeds_path = Path(state_dir) / "knowledge_base_seeds.json"
    dataset_name = "ucf" if is_ucf else "xd"
    seeds = load_semantic_seeds(seeds_path, dataset_name)
    discovered = discover_ucf_categories(annotation) if is_ucf else discover_xd_categories(annotation)
    missing = discovered - set(seeds)
    if missing:
        raise KnowledgeBaseError(
            f"missing semantic seed for discovered {dataset_name} labels: {sorted(missing)}"
        )
    if state_path.exists():
        try:
            knowledge_base = KnowledgeBase.load(state_path)
        except KnowledgeBaseError:
            knowledge_base = KnowledgeBase(
                ucf_paths=annotation if is_ucf else (),
                xd_paths=annotation if is_xd else (),
                state_path=state_path,
                seeds=seeds,
                require_seeds=True,
            )
        else:
            if is_ucf:
                knowledge_base.add_ucf_paths(annotation, seeds=seeds, require_seeds=True)
            else:
                knowledge_base.add_xd_paths(annotation, seeds=seeds, require_seeds=True)
    elif is_ucf:
        knowledge_base = KnowledgeBase(
            ucf_paths=annotation, state_path=state_path,
            seeds=seeds, require_seeds=True,
        )
    else:
        knowledge_base = KnowledgeBase(
            xd_paths=annotation, state_path=state_path,
            seeds=seeds, require_seeds=True,
        )
    knowledge_base.save()
    return knowledge_base


__all__ = [
    "DEFAULT_NEGATIVE_TEMPLATE",
    "DEFAULT_POSITIVE_TEMPLATE",
    "KnowledgeBase",
    "KnowledgeBaseError",
    "LearnerDecision",
    "OptimizerChange",
    "OptimizerProtocolError",
    "OptimizerUpdate",
    "PromptPair",
    "atomic_save_json",
    "discover_categories",
    "discover_ucf_categories",
    "discover_ucf_labels",
    "discover_xd_categories",
    "discover_xd_labels",
    "extract_ucf_category",
    "extract_xd_categories",
    "initialize_prompt_pairs",
    "_is_xd_annotation",
    "allowed_labels_for_batch",
    "build_update_scope",
    "load_or_create_knowledge_base",
    "load_semantic_seeds",
    "parse_learner_decision",
    "parse_optimizer_json",
    "parse_optimizer_update",
    "render_all",
]
