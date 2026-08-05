from __future__ import annotations

import json


def build_validation_prompt(
    trial_id: str,
    eligibility_text: str,
    extracted_rows: list[dict],
) -> str:
    rows_json = json.dumps(extracted_rows, ensure_ascii=False, indent=2)
    return f"""
You are an independent clinical-trial eligibility extraction auditor.

Audit trial {trial_id}. Work directly from the complete source eligibility text.
First enumerate every atomic inclusion and exclusion requirement yourself. Then
compare that independent inventory with the extracted rows. Do not assume an
extracted row is correct merely because it contains text copied from the source.

Classify each independently identified source criterion as:
- covered: fully represented by one or more extracted rows
- partially_covered: represented, but an important concept, value, operator,
  unit, temporal restriction, exception, or qualifier is missing
- uncovered: not represented

Classify every extracted row as:
- supported: fully grounded in the source text
- weakly_supported: related to the source, but materially distorted, overbroad,
  incorrectly valued, or assigned to the wrong inclusion/exclusion section
- unsupported: made up or not grounded in the source

Return ONLY one valid JSON object with this exact structure:
{{
  "source_criteria": [
    {{
      "criteria_type": "Inclusion or Exclusion",
      "source_clause": "exact or minimally trimmed source wording",
      "coverage_status": "covered|partially_covered|uncovered",
      "matched_row_indices": [0],
      "missing_concepts": [],
      "explanation": "brief evidence-based explanation",
      "confidence": 0.0
    }}
  ],
  "extracted_rows": [
    {{
      "row_index": 0,
      "support_status": "supported|weakly_supported|unsupported",
      "source_clause": "supporting source wording or null",
      "explanation": "brief evidence-based explanation",
      "confidence": 0.0
    }}
  ],
  "summary": "concise overall assessment",
  "quality_notes": ["specific actionable observation"]
}}

Rules:
1. Use only row_index values present in the supplied rows.
2. Include every supplied extracted row exactly once in extracted_rows.
3. Split compound source statements into atomic clinical requirements when they
   would normally require separate extracted rows.
4. Do not treat formatting headings as criteria.
5. A stricter or looser threshold than the source is not fully supported.
6. Preserve negation and inclusion/exclusion direction.
7. Confidence must be between 0 and 1.

SOURCE ELIGIBILITY TEXT:
{eligibility_text}

EXTRACTED ROWS:
{rows_json}
""".strip()


def build_chat_prompt(report_context: str, question: str) -> str:
    return f"""
You are discussing an extraction-integrity audit with a reviewer. Answer only
from the audit context below. Be concise, cite trial IDs and row indices where
useful, and clearly distinguish model judgments from deterministic metrics. If
the context cannot answer the question, say so.

AUDIT CONTEXT:
{report_context}

REVIEWER QUESTION:
{question}
""".strip()
