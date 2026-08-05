# Modern AutoCrit

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Desktop-Tkinter-orange)
![LLM](https://img.shields.io/badge/LLM-Multi--Provider-purple)
![Status](https://img.shields.io/badge/Status-Active%20Development-success)

## Overview

`modern-autocrit` is a modernization of the [AutoCriteria](https://pubmed.ncbi.nlm.nih.gov/37952206/) clinical trial eligibility criteria extraction workflow.

The goal is to refactor the original AutoCriteria-style pipeline into a cleaner, standalone application that uses the modern OpenAI SDK, removes legacy LangChain dependencies, and adds built-in tools for semantic deduplication and terminology normalization.

Modern AutoCrit is a standalone desktop application for extracting structured eligibility criteria from ClinicalTrials.gov XML files using Large Language Models (LLMs). The application provides an end-to-end workflow for configuring models, running the extraction pipeline, reviewing outputs, and performing quality assurance through anomaly detection and analytics.

The application currently consists of four primary modules:

---

### Run Pipeline

Execute the complete extraction pipeline from a graphical interface.

Features include:

- Select a folder containing ClinicalTrials.gov XML files
- Choose an output directory
- Run extraction without using the command line
- Real-time progress bar
- Live logging window
- Pipeline status updates
- Automatic generation of structured Excel outputs

<p align="center">
  <img src="figs/autocrit_run.png" width="900">
</p>

--- 

### ⚙️ Settings

Configure the LLM provider and extraction parameters before running the pipeline.

Current functionality includes:

- Select LLM provider (OpenAI, Anthropic, Gemini, Ollama)
- Select model
- Configure API credentials
- Adjust temperature and maximum output tokens
- Configure chunk size and overlap
- Enable or disable terminology normalization
- Enable or disable semantic deduplication
- Configure cost warning thresholds
- Test provider/model availability directly from the application

<p align="center">
  <img src="figs/autocrit_settings.png" width="900">
</p>

---

### Dictionary Management

Inspect and maintain terminology normalization dictionaries used during extraction.

Current capabilities include:

- View entity mappings
- View attribute mappings
- Search existing mappings
- Edit mappings
- Add new terminology
- Save updated dictionaries without modifying source code

This allows domain experts to continually improve terminology normalization as new trials are processed.

<p align="center">
  <img src="figs/autocrit_dictionary.png" width="900">
</p>

---

### Analytics & Quality Control

Review extraction outputs and perform automated quality assessment.

Current analytics include:

- Pipeline summary statistics
- Cost estimates
- Preview of extracted criteria
- Preview of unmapped terms
- Automated anomaly detection
- Quality-control reports for suspicious measurements and terminology
- Quick access to generated Excel outputs

Anomaly detection currently identifies:

- Implausible numeric measurements
- Unit mismatches
- Suspicious dosage values
- Placeholder or malformed terminology
- Missing or incomplete extracted values

<p align="center">
  <img src="figs/autocrit_analytics.png" width="900">
</p>

#### Integrity Validation

The **Integrity** tab performs an independent, LLM-assisted audit of one trial
or a group of trials. Select the source XML folder and extraction workbook, then
optionally enter specific NCT IDs. The validator inventories the source criteria,
flags missed or partially extracted requirements, identifies unsupported or
potentially fabricated rows, computes estimated precision/recall/F1 and coverage
metrics, and saves a multi-sheet Excel report. Its findings are review aids rather
than clinical ground truth; the chat panel supports follow-up questions grounded
in the completed audit. A progress bar tracks completed trials, and Stop prevents
additional trials from starting after the current model request finishes.

<p align="center">
  <img src="figs/autocrit_integrity.png" width="900">
</p>

---

## Goals

- Modernize the AutoCriteria extraction workflow.
- Replace legacy LangChain-based components with the modern OpenAI SDK.
- Provide a standalone desktop application interface.
- Support clinical trial eligibility criteria extraction from ClinicalTrials.gov records.
- Add semantic deduplication for repeated or overlapping extracted criteria.
- Add terminology normalization for mapping raw extracted terms to canonical concepts. Unmapped concepts get returned as well
- Provide configurable settings for output paths, model selection, API key management, and dictionary mappings.
- Deterministic mapping for terminology and semantic cleaning
- Multi-Model support and testing of integration

## Planned Features

- ClinicalTrials.gov trial download support.
- Modern eligibility criteria extraction pipeline.
- Structured criteria output.
- Cost monitoring utilities.
- Agentic Benchmarking and Evaluation of extraction

---

## Requirements

- Python 3.12 (recommended)
- OpenAI API key (or another supported LLM provider)
- macOS, Linux, or Windows

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/modern-autocrit.git
cd modern-autocrit
```

---

### 2. Create a Virtual Environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

---

### 3. Install Dependencies

Upgrade pip (recommended):

```bash
python -m pip install --upgrade pip
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

### 4. Configure API Access

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_api_key_here
```

Alternatively, API keys can be entered from the **Settings** tab when using the desktop application.

---

## Running Modern AutoCrit

### Download trial XML files

Search ClinicalTrials.gov and download a chosen number of matching trials:

```bash
python data_download.py "lung cancer AND recruiting" 25
```

Files are saved to `data/raw/xml_trials/` by default. Requests for 100 or more
trials require confirmation; use `--yes` for an intentional non-interactive
download. Run `python data_download.py --help` for all options.

### Update terminology dictionaries

After the pipeline creates `outputs/modern_autocrit_output.xlsx`, add newly
observed entity, attribute, and disease terminology to the normalization maps:

```bash
python dict_mapping.py
```

Use `--input path/to/output.xlsx` for another workbook. Existing mappings are
never overwritten. To restore all three dictionaries to their original
repository versions, run:

```bash
python dict_mapping.py --reset
```

### Option 1 — Desktop Application (Recommended)

Launch the full graphical application:

```bash
python main.py
```

The application provides five tabs:

- **Run** – Execute the extraction pipeline.
- **Settings** – Configure the API provider, model, and extraction settings.
- **Dictionary** – Review and edit terminology mappings.
- **Analytics** – Review extraction summaries, unmapped terms, and anomaly reports.
- **Integrity** – Audit extracted rows against the source trial criteria.

---

### Option 2 — Backend Pipeline Only

Run the extraction pipeline directly without launching the GUI:

```bash
python run_pipeline.py
```

This reads the configured XML directory, processes all trials, and writes the results to the `outputs/` directory.

---

## Input Data

Place ClinicalTrials.gov XML files inside:

```text
data/xml_trials/
```

Example:

```text
data/
└── xml_trials/
    ├── NCT00114192.xml
    ├── NCT00895128.xml
    └── ...
```

---

## Output Files

Successful runs generate:

```text
outputs/
├── modern_autocrit_output.xlsx
├── modern_autocrit_output_unmapped_terms.xlsx
├── modern_autocrit_output_anomalies.xlsx
└── integrity_validation.xlsx
```

where:

- **modern_autocrit_output.xlsx** contains the extracted structured eligibility criteria.
- **modern_autocrit_output_unmapped_terms.xlsx** lists attributes that could not be normalized.
- **modern_autocrit_output_anomalies.xlsx** contains automatically flagged measurement, terminology, and quality-control anomalies for manual review.
- **integrity_validation.xlsx** contains source-coverage findings, extraction-support findings, quality notes, validation metrics, and run status from the Integrity audit.
