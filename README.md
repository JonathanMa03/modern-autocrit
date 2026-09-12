# Modern AutoCrit: Automated Eligibility Criteria Extraction

A local browser application that retrieves clinical trial protocols and converts
free-text eligibility criteria into reviewed, computational Entity–Attribute–Value
(EAV) records.

## Features

- ClinicalTrials.gov API retrieval by NCT ID
- deterministic inclusion/exclusion source segmentation
- LLM-assisted atomic eligibility decomposition
- governed EAV terminology retrieval and normalization
- structured categorical values and numerical intervals
- mapping review, Attribute ID review, and semantic reconciliation
- recoverable jobs, atomic persistence, checkpoints, and result caching
- numerical clinical-distance and retained-relevance calculations (ToDo)
- searchable controlled EAV library

## Installation

Python 3.12 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Create a project-root `.env` file:

```text
OPENAI_API_KEY=your_api_key_here
```

## Run

```bash
python main.py
```

The app opens at [http://127.0.0.1:8765](http://127.0.0.1:8765). To prevent
automatic browser launch or use another port:

```bash
python main.py --no-browser --port 9000
```

Enter an NCT ID and leave the criteria box blank to retrieve the registered
protocol. Pasted criteria take precedence and also support draft protocols or
non-ClinicalTrials.gov sources.

## Architecture

```text
main.py
├── webapp/                         browser server and static interface
└── backend/
    ├── criteria_processor/         segmentation, extraction, EAV, review, recovery
    ├── ctg_parser/                 ClinicalTrials.gov API models and parsing
    ├── services/                   registry, LLM, job, and extraction adapters
    ├── statistics/                 clinical-distance models
    └── schemas/                    application configuration
```

The governed terminology sources are:

- `config/eligibility_eva_library.json` for the active eligibility workflow
- `config/attribute_library.xlsx` for workbook-based normalization workflows

Local jobs, source records, checkpoints, audits, and structured LLM call records
are stored under `runtime/eligcrit/` and ignored by Git.

## Testing

```bash
python -m pytest -q
```

Extraction results are model-assisted review artifacts, not clinical ground
truth. Proposed terminology is not silently merged into the governed library.
