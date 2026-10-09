"""Chia-inspired protocol-to-structured-criterion extraction pipeline.

The module name is retained for import compatibility, but output is no longer
a flat EAV record. Each criterion separates its domain/entity, value fields,
semantic constructs, Boolean logic, and verbatim provenance for downstream use.
"""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any

DOMAINS = ("observation", "condition", "person", "device", "drug", "visit", "procedure", "measurement")
LOGIC = ("standalone", "and", "or")
COMPARATOR = re.compile(
    r"(?P<comparator>>=|<=|>|<|=|≥|≤)\s*"
    r"(?P<value>[+-]?\d+(?:\.\d+)?(?:\s*/\s*[+-]?\d+(?:\.\d+)?)?)"
    r"(?P<unit>\s*(?:msec|ms|sec(?:ond)?s?|mmHg|mg/dL|mmol/L|µmol/L|umol/L|"
    r"kg/m2|kg/m²|IU/L|U/L|cells?/uL|mL/min(?:/1\.73m2)?|years?|months?|weeks?|days?|%))?", re.I,
)
RANGE = re.compile(
    r"(?:(?:between|from|aged?|ages?)\s*)?(?P<lower>\d+(?:\.\d+)?)\s*"
    r"(?:to|through|[-–—])\s*(?P<upper>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>years?(?:\s+old|\s+of\s+age)?|months?|weeks?|days?|kg/m2|kg/m²|mmHg|mg/dL|mmol/L|%)?", re.I,
)
BETWEEN = re.compile(
    r"between\s+(?P<lower>\d+(?:\.\d+)?)\s+and\s+(?P<upper>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>years?(?:\s+old|\s+of\s+age)?|months?|weeks?|days?|kg/m2|kg/m²|mmHg|mg/dL|mmol/L|%)?", re.I,
)
DOUBLE_COMPARATOR = re.compile(
    r"(?:>=|≥)\s*(?P<lower>\d+(?:\.\d+)?)\s*(?:and|,)?\s*(?:<=|≤)\s*(?P<upper>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>years?|months?|weeks?|days?|kg/m2|kg/m²|mmHg|mg/dL|mmol/L|%)?", re.I,
)
NEGATION = re.compile(
    r"(?:^|[;:,.]\s*)(?:no|not|without|negative\s+for|absence\s+of|free\s+of|never)\b|"
    r"\bno\s+(?:known\s+)?history\s+of\b", re.I,
)
REPETITION = re.compile(
    r"\b(?:on\s+)?(?P<count>one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(?P<label>occasions?|times?|measurements?|visits?)\b", re.I,
)
TEMPORAL_PATTERNS = (
    re.compile(r"\bhistory\s+of\b", re.I),
    re.compile(r"\bwithin\s+(?:the\s+)?(?:past|previous|last)?\s*\d+\s+(?:hours?|days?|weeks?|months?|years?)(?:\s+(?:before|prior to|after)\s+[^,.;]+)?", re.I),
    re.compile(r"\b(?:for|during)\s+(?:at least|more than|less than|>|<|≥|≤)?\s*\d+\s+(?:hours?|days?|weeks?|months?|years?)\b", re.I),
    re.compile(r"\b(?:in|over)\s+(?:the\s+)?(?:past|previous|last)?\s*\d+\s+(?:hours?|days?|weeks?|months?|years?)\b", re.I),
    re.compile(r"\b(?:at least|more than|less than)\s+\d+\s+(?:hours?|days?|weeks?|months?|years?)\b", re.I),
)
TERMINOLOGY_PATH = Path(__file__).resolve().parents[1] / "config" / "terminology_normalization.json"


def terminology_aliases() -> dict[str, str]:
    try:
        payload = json.loads(TERMINOLOGY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(k).casefold().strip(): str(v).strip() for k, v in (payload.get("aliases") or {}).items()}


def terminology_config() -> dict[str, Any]:
    try:
        return json.loads(TERMINOLOGY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


ROW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"rows": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "criterion_type": {"type": "string", "enum": ["inclusion", "exclusion"]},
            "domain": {"type": "string", "enum": list(DOMAINS)},
            "entity": {"type": "string", "minLength": 1},
            "comparator": {"type": ["string", "null"], "enum": [">", ">=", "<", "<=", "=", None]},
            "value": {"type": ["string", "number", "integer", "null"]},
            "unit": {"type": ["string", "null"]},
            "negated": {"type": "boolean"},
            "temporal": {"type": ["string", "null"]},
            "repetition": {"type": ["string", "null"]},
            "qualifier": {"type": ["string", "null"]},
            "logical_operator": {"type": "string", "enum": list(LOGIC)},
            "lower_bound": {"type": ["number", "null"]},
            "lower_inclusive": {"type": ["boolean", "null"]},
            "upper_bound": {"type": ["number", "null"]},
            "upper_inclusive": {"type": ["boolean", "null"]},
            "extraction_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "source": {"type": "string", "minLength": 1},
        },
        "required": ["criterion_type", "domain", "entity", "comparator", "value", "unit", "negated", "temporal", "repetition", "qualifier", "logical_operator", "lower_bound", "lower_inclusive", "upper_bound", "upper_inclusive", "extraction_confidence", "source"],
        "additionalProperties": False,
    }}},
    "required": ["rows"], "additionalProperties": False,
}


def canonical_term(value: str) -> str:
    text = re.sub(r"[_\s]+", " ", str(value or "")).strip(" .,:;-")
    config = terminology_config()
    aliases = {str(k).casefold().strip(): str(v).strip() for k, v in (config.get("aliases") or {}).items()}
    exact_only = {str(item).casefold() for item in config.get("exact_only_aliases", [])}
    exact = aliases.get(text.casefold())
    if exact:
        return exact
    protected: dict[str, str] = {}
    for index, (source, canonical) in enumerate(sorted(aliases.items(), key=lambda x: len(x[0]), reverse=True)):
        if source in exact_only:
            continue
        marker = f"zzcanonicaltoken{index}zz"
        updated, count = re.subn(rf"(?<!\w){re.escape(source)}(?!\w)", marker, text, flags=re.I)
        if count:
            text, protected[marker] = updated, canonical
    if text and len(text) <= 8 and re.fullmatch(r"[A-Za-z0-9-]+", text):
        return text.upper()
    text = text.casefold()
    for marker, canonical in protected.items():
        text = text.replace(marker, canonical)
    return text


def infer_domain(entity: str, source: str, has_comparator: bool) -> str:
    text = f"{entity} {source}".casefold()
    for alias, domain in sorted((terminology_config().get("domain_aliases") or {}).items(), key=lambda item: len(item[0]), reverse=True):
        if domain in DOMAINS and re.search(rf"(?<!\w){re.escape(str(alias))}(?!\w)", text, re.I):
            return domain
    if re.search(r"\b(age|sex|gender|male|female|pregnan|body mass index|bmi)\b", text): return "person"
    if re.search(r"\b(drug|medication|prednisone|estrogen|therapy|dose|inhibitor)\b", text): return "drug"
    if re.search(r"\b(surgery|procedure|resection|transplant|dialysis|imaging|biopsy)\b", text): return "procedure"
    if re.search(r"\b(device|implant|pacemaker|prosthe)\b", text): return "device"
    if re.search(r"\b(visit|screening|follow-up|hospitali[sz]ation)\b", text): return "visit"
    if has_comparator or re.search(r"\b(level|count|pressure|score|index|test|laboratory|hba1c)\b", text): return "measurement"
    if re.search(r"\b(history|diagnos|disease|cancer|syndrome|failure|impairment|migraine|thromb|embol)\b", text): return "condition"
    return "observation"


def _temporal(source: str) -> str | None:
    found = [m.group(0).strip() for pattern in TEMPORAL_PATTERNS for m in pattern.finditer(source)]
    return "; ".join(dict.fromkeys(found)) or None


def _repetition(source: str) -> str | None:
    match = REPETITION.search(source)
    return match.group(0).strip() if match else None


def semantic_relations(*, comparator: str | None, negated: bool, temporal: str | None, repetition: str | None, qualifier: str | None, logical_operator: str) -> list[str]:
    """Project the compact record into Chia-style relation labels."""
    relations = []
    if comparator: relations.append("HAS_VALUE")
    if negated: relations.append("HAS_NEGATION")
    if temporal: relations.append("HAS_TEMPORAL")
    if repetition: relations.append("HAS_MULTIPLIER")
    if qualifier: relations.append("HAS_QUALIFIER")
    if logical_operator in {"and", "or"}: relations.append(logical_operator.upper())
    return relations


def _number(value: str) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


def interval_notation(lower: int | float | None, lower_inclusive: bool | None, upper: int | float | None, upper_inclusive: bool | None) -> str | None:
    if lower is None and upper is None:
        return None
    return f"{'[' if lower_inclusive else '('}{'-inf' if lower is None else lower}, {'+inf' if upper is None else upper}{']' if upper_inclusive else ')'}"


def deterministic_criterion(*, source: str, criterion_type: str, entity: str = "", domain: str = "", comparator: str | None = None, value: Any = None, unit: str | None = None, negated: bool | None = None, qualifier: str | None = None, temporal: str | None = None, repetition: str | None = None, logical_operator: str = "standalone", lower_bound: Any = None, lower_inclusive: bool | None = None, upper_bound: Any = None, upper_inclusive: bool | None = None) -> dict[str, Any]:
    """Enforce literal comparison fields and retain semantic constructs."""
    raw = str(source or "").strip()
    normalized_raw = re.sub(r"\\(?=[<>])", "", raw).replace("≥", ">=").replace("≤", "<=")
    normalized_entity = re.sub(r"\\(?=[<>])", "", str(entity or "")).strip()
    range_match = DOUBLE_COMPARATOR.search(normalized_raw) or BETWEEN.search(normalized_raw) or RANGE.search(normalized_raw)
    entity_position = normalized_raw.casefold().find(normalized_entity.casefold()) if normalized_entity else -1
    match = COMPARATOR.search(
        normalized_raw,
        entity_position + len(normalized_entity) if entity_position >= 0 else 0,
    )
    if match is None:
        match = COMPARATOR.search(normalized_raw)
    source_negated = bool(NEGATION.search(raw))
    proposed_negated = bool(negated) if negated is not None else False
    final_negated = source_negated or proposed_negated
    final_comparator: str | None = None
    final_value: str = "absent" if final_negated else "present"
    final_unit: str | None = None
    value_kind = "categorical"
    measured = normalized_entity
    final_lower = final_upper = None
    final_lower_inclusive = final_upper_inclusive = None
    if range_match:
        final_lower = _number(range_match.group("lower")); final_upper = _number(range_match.group("upper"))
        final_lower_inclusive = True; final_upper_inclusive = True
        final_value = f"{final_lower} to {final_upper}"
        final_unit = (range_match.groupdict().get("unit") or "").strip() or (str(unit).strip() if unit else None)
        if not final_unit and re.search(r"\b(?:aged?|ages?)\b", normalized_raw, re.I): final_unit = "years"
        if final_unit and final_unit.casefold().startswith("year"): final_unit = "years"
        measured = normalized_entity
        if not measured or re.search(r"\d", measured):
            measured = normalized_raw[:range_match.start()].strip(" \t:;,-()[]") or "age"
        measured = re.sub(r"^(?:aged?|ages?)\b", "age", measured, flags=re.I).strip()
        value_kind = "interval"
    elif match:
        final_comparator = match.group("comparator").replace("≥", ">=").replace("≤", "<=")
        measured = re.split(r"(?:>=|<=|>|<|=|≥|≤)", normalized_entity, maxsplit=1)[0].strip(" \t:;,-()[]")
        if not measured: measured = normalized_raw[:match.start()].strip(" \t:;,-()[]")
        if "(" in measured and not measured.endswith(")"): measured = measured.rsplit("(", 1)[-1].strip()
        final_value = re.sub(r"\s+", "", match.group("value"))
        final_unit = (match.group("unit") or "").strip() or (str(unit).strip() if unit else None)
        value_kind = "numerical"
        if "/" not in match.group("value"):
            bound = _number(match.group("value"))
            if final_comparator in {">", ">="}: final_lower, final_lower_inclusive = bound, final_comparator == ">="
            elif final_comparator in {"<", "<="}: final_upper, final_upper_inclusive = bound, final_comparator == "<="
            elif final_comparator == "=": final_lower = final_upper = bound; final_lower_inclusive = final_upper_inclusive = True
    elif comparator in {">", ">=", "<", "<=", "="} and value is not None:
        final_comparator = comparator
        final_value = str(value).strip()
        final_unit = str(unit).strip() if unit else None
        value_kind = "numerical"
        try:
            bound = _number(str(value))
            if final_comparator in {">", ">="}: final_lower, final_lower_inclusive = bound, final_comparator == ">="
            elif final_comparator in {"<", "<="}: final_upper, final_upper_inclusive = bound, final_comparator == "<="
            elif final_comparator == "=": final_lower = final_upper = bound; final_lower_inclusive = final_upper_inclusive = True
        except ValueError:
            pass
    else:
        measured = measured or raw
        measured = re.sub(r"^(?:must\s+have|patients?\s+with|participants?\s+with|history\s+of|no\s+history\s+of|without|exclude(?:d)?\s+)", "", measured, flags=re.I).strip(" .,:;-")
        proposed_value = str(value).strip() if value is not None else ""
        if proposed_value and not final_negated:
            final_value = proposed_value
    if final_lower is None and lower_bound is not None:
        try: final_lower = _number(str(lower_bound))
        except ValueError: pass
        final_lower_inclusive = lower_inclusive
    if final_upper is None and upper_bound is not None:
        try: final_upper = _number(str(upper_bound))
        except ValueError: pass
        final_upper_inclusive = upper_inclusive
    if final_lower is not None and final_upper is not None: value_kind = "interval"
    canonical = canonical_term(measured)
    final_temporal = temporal or _temporal(normalized_raw)
    final_repetition = repetition or _repetition(normalized_raw)
    final_qualifier = str(qualifier).strip() if qualifier else None
    final_logic = logical_operator if logical_operator in LOGIC else "standalone"
    return {
        "criterion_type": criterion_type,
        "domain": domain if domain in DOMAINS else infer_domain(canonical, raw, bool(match)),
        "entity": canonical,
        "comparator": final_comparator,
        "value": final_value,
        "unit": final_unit,
        "value_kind": value_kind,
        "lower_bound": final_lower,
        "lower_inclusive": final_lower_inclusive,
        "upper_bound": final_upper,
        "upper_inclusive": final_upper_inclusive,
        "interval": interval_notation(final_lower, final_lower_inclusive, final_upper, final_upper_inclusive),
        "negated": final_negated,
        "temporal": final_temporal,
        "repetition": final_repetition,
        "qualifier": final_qualifier,
        "logical_operator": final_logic,
        "relations": semantic_relations(comparator=final_comparator or ("interval" if final_lower is not None or final_upper is not None else None), negated=final_negated, temporal=final_temporal, repetition=final_repetition, qualifier=final_qualifier, logical_operator=final_logic),
        "source": raw,
    }


def deterministic_eav(*, source: str, criterion_type: str, entity: str = "") -> dict[str, Any]:
    """Compatibility alias for code transitioning from the former EAV API."""
    return deterministic_criterion(source=source, criterion_type=criterion_type, entity=entity)


def extraction_prompt(eligibility_text: str) -> str:
    return extraction_prompt_for_items(segment_criteria(eligibility_text))


def extraction_prompt_for_items(items: list[dict[str, str]]) -> str:
    return (
        "Extract only from the supplied eligibility criteria and copy each source exactly. "
        "Return one row per atomic clinical concept. Split coordinated diseases, drugs, tests, "
        "and measurements into separate rows, retaining their shared source and and/or relation. "
        "Use Chia domains: observation, condition, person, device, drug, visit, procedure, or measurement. "
        "Entity is only the clinical concept name; never put comparator, value, unit, temporal phrase, "
        "repetition, qualifier, or negation in entity. Separate comparator, literal value, and unit. "
        "For bounded ranges, fill lower_bound, upper_bound, and both inclusive flags; for example, "
        "'aged 18 to 65' is entity age with bounds 18 and 65, both inclusive, and unit years. "
        "For nonnumeric criteria use value present or absent. Negated means the entity itself is absent; "
        "'diabetes not due to steroids' does not negate diabetes. Capture temporal restrictions, repetition "
        "such as 'on three occasions', material qualifiers/exceptions, and Boolean logic separately. "
        "Give extraction_confidence from 0 to 1 for faithfulness of each complete row; this is not human approval. "
        "Use null when a semantic field is not stated. Do not infer facts. INPUT:\n" + json.dumps(items, ensure_ascii=False)
    )


def _stable_id(prefix: str, *parts: Any) -> str:
    material = "\x1f".join(str(part or "").strip() for part in parts)
    return f"{prefix}-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:16]}"


def segment_criteria(text: str, *, protocol_id: str = "UNSPECIFIED", source_version: str = "unversioned") -> list[dict[str, Any]]:
    current = "inclusion"
    has_heading = bool(re.search(r"^\s*(?:\d+(?:\.\d+)*\s+)?(?:inclusion|exclusion)\s+criteria\s*:?.*$", str(text or ""), re.I | re.M))
    started, rows, active = not has_heading, [], None
    for raw in str(text or "").replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        if not line: continue
        heading = re.fullmatch(r"(?:\d+(?:\.\d+)*\s+)?\*{0,2}(inclusion|exclusion)(?:\s+criteria)?\s*:?\*{0,2}", line, re.I)
        if heading:
            current, started, active = heading.group(1).casefold(), True, None; continue
        if not started: continue
        item = re.match(r"^(?:[-*•]\s+|\d+[.)]\s+)(.*)$", line)
        if item:
            active = {"criterion_type": current, "source": item.group(1).strip()}; rows.append(active)
        elif has_heading and active is None and (COMPARATOR.search(line) or RANGE.search(line) or BETWEEN.search(line) or ";" in line):
            active = {"criterion_type": current, "source": line}; rows.append(active)
        elif has_heading and line.endswith(":"): active = None
        elif not has_heading:
            active = {"criterion_type": current, "source": line}; rows.append(active)
        elif active is not None and not re.match(r"^\d+(?:\.\d+)+\s+", line):
            active["source"] = f"{active['source']} {line}".strip()
    atomic: list[dict[str, Any]] = []
    for source_index, row in enumerate(rows, start=1):
        parent = row["source"]
        parent_id = _stable_id("statement", protocol_id, source_version, row["criterion_type"], source_index, parent)
        fragments = [x.strip() for x in re.split(r"\s*;\s*", parent) if x.strip()]
        previous = rows[source_index - 2]["source"] if source_index > 1 else ""
        following = rows[source_index]["source"] if source_index < len(rows) else ""
        context = "\n".join(part for part in (previous, parent, following) if part)
        for fragment in fragments:
            atomic.append({
                "criterion_type": row["criterion_type"], "source": fragment,
                "source_id": parent_id, "parent_statement_id": parent_id,
                "parent_statement": parent, "parent_source_index": source_index,
                "surrounding_context": context,
            })
    return atomic


def normalize_rows(
    payload: dict[str, Any], source_items: list[dict[str, Any]] | None = None, *,
    protocol_id: str = "UNSPECIFIED", source_version: str = "unversioned",
    source_uri: str | None = None, source_retrieved_at: str | None = None,
) -> list[dict[str, Any]]:
    proposed = payload.get("rows") or []
    source_types: dict[str, str] = {}; source_items_by_text: dict[str, dict[str, Any]] = {}
    if source_items is not None:
        source_types = {x["source"]: x["criterion_type"] for x in source_items}
        source_items_by_text = {x["source"]: x for x in source_items}
        reconciled, seen = [], set()
        for row in proposed:
            source = str(row.get("source") or "").strip()
            if source in source_types:
                reconciled.append({**row, "source": source, "criterion_type": source_types[source]}); seen.add(source)
        reconciled.extend(x for x in source_items if x["source"] not in seen)
        proposed = reconciled
    counts: dict[str, int] = {}
    for row in proposed:
        source = str(row.get("source") or "").strip(); counts[source] = counts.get(source, 0) + 1
    rows: list[dict[str, Any]] = []
    atom_counts: dict[str, int] = {}
    for index, row in enumerate(proposed, start=1):
        source = str(row.get("source") or "").strip()
        item = source_items_by_text.get(source, {})
        parent_id = str(item.get("parent_statement_id") or item.get("source_id") or _stable_id("statement", protocol_id, source_version, source))
        atom_counts[parent_id] = atom_counts.get(parent_id, 0) + 1
        atom_index = atom_counts[parent_id]
        logic = str(row.get("logical_operator") or "standalone").casefold()
        if counts.get(source, 0) > 1 and logic == "standalone": logic = "or" if re.search(r"\bor\b", source, re.I) else "and"
        normalized = deterministic_criterion(
            source=source, criterion_type=str(row.get("criterion_type") or "inclusion").casefold(),
            entity=str(row.get("entity") or ""), domain=str(row.get("domain") or ""),
            comparator=row.get("comparator"), value=row.get("value"), unit=row.get("unit"),
            negated=row.get("negated"),
            lower_bound=row.get("lower_bound"), lower_inclusive=row.get("lower_inclusive"),
            upper_bound=row.get("upper_bound"), upper_inclusive=row.get("upper_inclusive"),
            qualifier=row.get("qualifier"), temporal=row.get("temporal"), repetition=row.get("repetition"), logical_operator=logic,
        )
        normalized["criterion_id"] = _stable_id("criterion", protocol_id, source_version, parent_id, atom_index)
        normalized["row_id"] = normalized["criterion_id"]
        normalized["source_id"] = parent_id
        normalized["parent_statement_id"] = parent_id
        normalized["parent_statement"] = str(item.get("parent_statement") or source)
        normalized["atom_index"] = atom_index
        normalized["surrounding_context"] = str(item.get("surrounding_context") or source)
        normalized["protocol_id"] = protocol_id
        normalized["source_version"] = source_version
        normalized["source_uri"] = source_uri
        normalized["source_retrieved_at"] = source_retrieved_at
        confidence = row.get("extraction_confidence", 0.5)
        try: normalized["extraction_confidence"] = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError): normalized["extraction_confidence"] = 0.5
        normalized["human_review_status"] = "pending"
        normalized["reviewed_at"] = None
        rows.append(normalized)
    parent_counts: dict[str, int] = {}
    for row in rows:
        parent_counts[row["parent_statement_id"]] = parent_counts.get(row["parent_statement_id"], 0) + 1
    from backend.criterion_validation import readiness, validate_criterion
    for row in rows:
        row["parent_atom_count"] = parent_counts[row["parent_statement_id"]]
        row["boolean_group_id"] = row["parent_statement_id"]
        if row["parent_atom_count"] > 1 and row["logical_operator"] == "standalone":
            row["logical_operator"] = "or" if re.search(r"\bor\b", row["parent_statement"], re.I) else "and"
            row["relations"] = semantic_relations(
                comparator=row.get("comparator") or ("interval" if row.get("interval") else None),
                negated=bool(row.get("negated")), temporal=row.get("temporal"),
                repetition=row.get("repetition"), qualifier=row.get("qualifier"),
                logical_operator=row["logical_operator"],
            )
        row["validation_issues"] = validate_criterion(row)
        row["export_ready"], row["readiness_blockers"] = readiness(row)
    return rows
