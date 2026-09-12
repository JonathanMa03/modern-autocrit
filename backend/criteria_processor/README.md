# Eligibility criteria processor

This package contains the domain layer used by the browser application.

The active extraction flow is:

1. `segmentation.py` separates inclusion and exclusion source rows.
2. `breakdown_cli.py` performs resumable LLM-assisted atomic decomposition.
3. `eva_pipeline.py` retrieves governed terminology candidates and extracts EVA records.
4. `eva_review.py` reviews mapping and Attribute ID decisions.
5. `eva_library.py` validates and versions the controlled terminology library.
6. `eva_reconciliation*.py` detects and resolves proposed terminology conflicts.
7. `eva_recovery.py`, `eva_jobs.py`, and `atomic_io.py` provide audit recovery and safe persistence.
8. `value_normalization.py` and `numeric_intervals.py` represent computable values.

Application orchestration belongs in `backend/services/eligcrit_extraction.py`.
The package does not fetch ClinicalTrials.gov records or serve browser requests.
