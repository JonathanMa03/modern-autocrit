# Automated Eligibility Criteria Extraction

A local browser application that converts protocol eligibility text into
human-reviewed, MetricSpace-ready structured criteria.

## Workflow

1. Choose ClinicalTrials.gov ingestion by NCT ID, paste eligibility criteria,
   or upload a complete protocol PDF. PDF ingestion locates eligibility sections
   and excludes downstream SAP/SOA content from extraction.
2. Run structured extraction. A progress bar reports ingestion, extraction,
   canonical normalization, completion, or failure.
3. Sort and review Chia-inspired fields: inclusion/exclusion, clinical domain,
   entity, comparator, value, unit, negation, temporal scope, repetition,
   qualifier, Boolean logic, and source provenance.
4. Accept correct rows. Rejecting a row opens its editable fields; saving the
   correction marks it accepted.
5. Apply all reviewed rows. The application writes cumulative timestamped
   `criteriaLibrary_YYYYMMDD_HHMMSS_microseconds.json` and `.csv` snapshots under
   both `runtime/criteria/libraries/` and `config/`. Generated config snapshots are
   ignored by Git; the packaged base remains immutable.

Numerical restrictions retain their literal components. For example:

- `GPL > 50` becomes Domain `measurement`, Entity `GPL`, Comparator `>`,
  Value `50`.
- `blood pressure > 145/95 on three occasions` becomes Domain `measurement`,
  Entity `blood pressure`, Comparator `>`, Value `145/95`, and Repetition
  `on three occasions`.
- `aged 18 to 65` becomes Domain `person`, Entity `Age`, lower bound `18`
  inclusive, upper bound `65` inclusive, unit `years`, and interval `[18, 65]`.

Equivalent range forms such as `between 18 and 65 years`, `18–65 years of
age`, and `age >=18 and <=65 years` normalize to the same closed interval.
Single-sided comparisons also expose their corresponding open or closed bound,
which supports exact overlap, subset, superset, and disjointness calculations.

Categorical restrictions use the explicit values `present` and `absent`, with
negation stored separately. Temporal restrictions, repetition, qualifiers,
AND/OR relations, inclusion/exclusion, and verbatim source text remain distinct.
The distinction between clinical domains, value fields, semantic constructs,
and relations is adapted from the
[Chia clinical-trial eligibility corpus](https://www.nature.com/articles/s41597-020-00620-0).
The project uses a compact JSON/CSV projection rather than claiming full Chia
annotation-graph compatibility.

The packaged [base criteria library](config/base_criterion_library.json) is read-only.
Only timestamped generated snapshots receive reviewed additions.
Canonical aliases are governed in
[terminology_normalization.json](config/terminology_normalization.json).
That configuration includes the legacy attribute and disease maps, translates
the legacy entity-class map into Chia domains, and adds frequently encountered
oncology diseases, performance measures, laboratory terms, response concepts,
and biomarkers. Ambiguous short oncology abbreviations are exact-match only so
ordinary words such as “all” are not rewritten inside sentences.

The extractor retains the useful core of the original AutoCriteria workflow:
source-bounded prompting, separate inclusion/exclusion processing, clinical
entity-class guidance, one clinical concept per output row, and verbatim source
provenance. This implementation replaces its LangChain/pandas script with the
project's OpenAI SDK adapter, strict JSON schema, deterministic post-processing,
persisted job records, and mandatory human review.

## Teammate setup and execution

Prerequisites:

- Python 3.12 or newer
- an OpenAI API key with access to the model selected in
  `config/default_settings.json`
- internet access for model calls and ClinicalTrials.gov ingestion

Clone the project and enter its root directory:

```bash
git clone https://github.com/JonathanMa03/modern-autocrit.git
cd modern-autocrit
```

Create the environment and install dependencies on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create `.env` in the repository root:

```text
OPENAI_API_KEY=your_api_key_here
```

Start the backend and browser UI with one command:

```bash
python main.py
```

The command starts both the backend and frontend and normally opens
[http://127.0.0.1:8765](http://127.0.0.1:8765). Keep that terminal open while
using the application. Stop it with `Ctrl+C`.

Useful alternatives:

```bash
# Start without opening a browser automatically
python main.py --no-browser

# Use a different port if 8765 is occupied
python main.py --port 9000
```

The API key stays on the local backend and is never entered into the browser.

## Running an extraction

1. Open **Extraction / ingestion**.
2. Choose one input mode:
   - **ClinicalTrials.gov NCT ID** retrieves the registered eligibility text.
   - **Paste eligibility criteria** accepts local or draft inclusion/exclusion
     text.
   - **Upload full protocol PDF** accepts a text-based complete protocol and
     extracts only recognizable eligibility sections.
3. Select **Run extraction** and follow the progress log. It reports how many
   source criteria were located and how many have been processed.
4. In **Human review**, sort rows by their headers. Select ✓ to accept a row or
   ✕ to edit it. Saving a correction counts as acceptance.
5. After every row is reviewed, select **Apply reviewed criteria to library**.
6. Use **Criteria library** to search the immutable examples and latest reviewed
   trial criteria.

## Outputs

### Structured review rows

Each extracted row provides:

| Field | Meaning |
| --- | --- |
| `criterion_type` | Inclusion or exclusion polarity from the protocol section. |
| `domain` | Chia-style domain: observation, condition, person, device, drug, visit, procedure, or measurement. |
| `entity` | Canonical clinical concept without its threshold or modifier. |
| `comparator`, `value`, `unit` | Literal quantitative or categorical constraint components. |
| `lower_bound`, `upper_bound` | Nullable numerical endpoints for MetricSpace comparison. |
| `lower_inclusive`, `upper_inclusive` | Whether each endpoint is closed. |
| `interval` | Readable representation such as `[18, 65]` or `(18, +inf)`. |
| `negated` | Whether absence of the entity is required. |
| `temporal` | Time window separated from the entity. |
| `repetition` | Frequency or multiplicity, such as `on three occasions`. |
| `qualifier` | Severity, exception, or other clinically material modifier. |
| `logical_operator` | Standalone, AND, or OR relationship for concepts from a shared source. |
| `relations` | Derived Chia-style labels such as `HAS_VALUE` and `HAS_TEMPORAL`. |
| `source_id`, `source` | Stable provenance identifier and verbatim protocol evidence. |

### Files created

- `runtime/criteria/jobs/<job_id>.json` contains status, progress events,
  errors, and completed review rows for one submission.
- `runtime/criteria/libraries/criteriaLibrary_<datetime>.json` is the
  authoritative local reviewed snapshot.
- `runtime/criteria/libraries/criteriaLibrary_<datetime>.csv` is its tabular
  export.
- Matching JSON and CSV snapshots are written under `config/` so the currently
  reviewed library is available to the application. Generated snapshots are
  ignored by Git.
- `config/base_criterion_library.json` is the packaged, immutable seed library.
- `config/terminology_normalization.json` contains governed aliases, oncology
  terminology, exact-only abbreviations, and legacy-class-to-domain mappings.

The JSON snapshot uses schema `modern-autocrit.structured-criteria.v3`.
Historical `eavLibrary_*.json` files are read through an in-memory compatibility
upgrade and are not modified.

## Operational notes

- The PDF reader requires selectable text and does not perform OCR.
- A failed model batch currently fails the job; automatic checkpoint resume and
  partial continuation remain future work.
- Never commit `.env`, `runtime/`, uploaded protocols, or generated library
  snapshots.
- If startup reports “address already in use,” stop the prior process or use a
  different `--port`.
- Model output must be reviewed; it is not clinical ground truth.

## Architecture

```text
main.py                         launcher
webapp/server.py                HTTP API and static-file server
webapp/static/                  three-panel browser interface
backend/eav_pipeline.py         segmentation and Chia-inspired semantic normalization
backend/services/eav_workflow.py jobs, retries, progress, review, JSON/CSV snapshots
backend/services/ctgov_v2_service.py ClinicalTrials.gov v2 retrieval
backend/services/llm/           OpenAI SDK boundary
config/base_criterion_library.json immutable packaged seed library
runtime/criteria/               ignored local jobs and reviewed snapshots
```

## Testing

```bash
python -m pytest -q
```

Model output is always subject to human review and is not clinical ground truth.
