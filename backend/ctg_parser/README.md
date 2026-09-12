# ClinicalTrials.gov parser

This package is the target-agnostic ClinicalTrials.gov parsing layer.

Current boundary:

1. Accept raw ClinicalTrials.gov API v2 study JSON.
2. Preserve structured registry fields such as conditions, arms, interventions,
   outcomes, eligibility text, demographic constraints, dates, sponsors, and
   source URLs.
3. Return validated Pydantic models that can be consumed by downstream
   criteria-processing workflows.

It intentionally does not contain disease-specific search logic, trial
filtering, target profiles, LLM extraction, PDF ingestion, or export code.
