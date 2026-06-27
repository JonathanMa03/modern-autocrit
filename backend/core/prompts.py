"""
Prompt templates for Modern AutoCrit.
"""

from __future__ import annotations


ENTITY_TYPES = [
    "Demographic",
    "Diagnosis",
    "Comorbidity",
    "Biomarker",
    "Lab Test",
    "Vital",
    "Score",
    "Treatment History",
    "Medication",
    "Procedure",
    "Contraceptive",
    "Survival",
    "Consent",
    "Other",
]


def build_extraction_prompt(
    criteria_text: str,
    criteria_type: str,
    disease_context: str | None = None,
) -> str:
    disease_context = disease_context or "the disease or condition under study"

    return f"""
You are an expert clinical trial eligibility criteria extraction system.

Your goal is to extract EVERY eligibility criterion from the provided clinical trial {criteria_type} criteria text.

Disease context: {disease_context}

Return ONLY valid JSON.
Return a JSON array.
Do not use markdown.
Do not explain your answer.
If no eligibility criteria are present, return [].

Each JSON object must contain exactly these keys:
- Entity
- Attribute
- Value
- Condition
- Sentence

Allowed Entity values:
{ENTITY_TYPES}

Extract all of the following if present:
- age
- gender
- education
- consent requirements
- language requirements
- diagnosis criteria
- disease stage
- disease status
- measurable disease
- biomarkers
- mutations
- lab tests
- organ function requirements
- marrow function requirements
- vitals
- ECOG performance status
- Karnofsky score
- other clinical scores
- prior treatments
- prior therapies
- chemotherapy
- immunotherapy
- targeted therapy
- radiation therapy
- surgery
- medications or drugs
- procedures
- imaging requirements
- comorbidities
- infections
- cardiovascular disease
- autoimmune disease
- psychiatric conditions
- allergies or hypersensitivity
- contraindications
- pregnancy or lactation
- contraception requirements
- life expectancy
- ability to comply with study procedures
- any other eligibility requirement

Field definitions:
- Entity: one category from the allowed entity values.
- Attribute: the specific clinical concept, test, disease, treatment, drug, biomarker, score, or requirement.
- Value: numeric threshold, yes/no, allowed/not allowed, required value, or status.
- Condition: temporal restriction, exception, cohort qualifier, modifier, or patient subgroup detail.
- Sentence: exact source sentence or phrase from the input text.

Important extraction rules:
1. Extract exhaustively. Do not summarize.
2. Create one row per criterion.
3. Preserve the source sentence.
4. If a sentence contains multiple criteria, create multiple rows.
5. If a lab sentence lists multiple labs, create one row per lab.
6. If a disease sentence lists multiple diseases, create one row per disease.
7. If a treatment sentence lists multiple treatments, create one row per treatment.
8. Do not invent criteria.
9. Do not create rows saying "the sentence does not mention..." or "the text does not provide...".
10. Prefer short atomic attributes, but do not omit valid criteria.
11. The Attribute should usually be a concise clinical concept, not the entire sentence.
12. Put modifiers, exceptions, and cohort-specific details in Condition.
13. For inclusion criteria, Value usually means required/allowed status.
14. For exclusion criteria, Value usually means presence of that attribute excludes the patient.

Examples:

Input sentence:
"Patients must have ANC >= 1500/mm3 and platelet count >= 100000/mm3."

Output rows:
[
  {{
    "Entity": "Lab Test",
    "Attribute": "absolute neutrophil count",
    "Value": ">= 1500/mm3",
    "Condition": "",
    "Sentence": "Patients must have ANC >= 1500/mm3 and platelet count >= 100000/mm3."
  }},
  {{
    "Entity": "Lab Test",
    "Attribute": "platelet count",
    "Value": ">= 100000/mm3",
    "Condition": "",
    "Sentence": "Patients must have ANC >= 1500/mm3 and platelet count >= 100000/mm3."
  }}
]

Input sentence:
"Patients with HIV, hepatitis B, or hepatitis C infection are excluded."

Output rows:
[
  {{
    "Entity": "Comorbidity",
    "Attribute": "HIV infection",
    "Value": "Yes",
    "Condition": "",
    "Sentence": "Patients with HIV, hepatitis B, or hepatitis C infection are excluded."
  }},
  {{
    "Entity": "Comorbidity",
    "Attribute": "hepatitis B infection",
    "Value": "Yes",
    "Condition": "",
    "Sentence": "Patients with HIV, hepatitis B, or hepatitis C infection are excluded."
  }},
  {{
    "Entity": "Comorbidity",
    "Attribute": "hepatitis C infection",
    "Value": "Yes",
    "Condition": "",
    "Sentence": "Patients with HIV, hepatitis B, or hepatitis C infection are excluded."
  }}
]

Input sentence:
"Ability to understand and willingness to sign a written informed consent document."

Output rows:
[
  {{
    "Entity": "Consent",
    "Attribute": "written informed consent",
    "Value": "Yes",
    "Condition": "ability to understand and willingness to sign",
    "Sentence": "Ability to understand and willingness to sign a written informed consent document."
  }}
]

Now extract criteria from this text:

{criteria_text}
""".strip()


def build_temporal_prompt(
    sentence: str,
    attribute: str,
) -> str:
    return f"""
Extract the specific temporal restriction associated with the attribute.

Return only valid JSON with one key: "temporal".
If no temporal restriction exists, return {{"temporal": ""}}.

Attribute:
{attribute}

Sentence:
{sentence}
""".strip()