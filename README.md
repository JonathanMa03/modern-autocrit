# modern-autocrit

## Overview

modern-autocrit is a modernization of the AutoCriteria clinical trial eligibility criteria extraction workflow.

The goal is to refactor the original AutoCriteria-style pipeline into a cleaner, standalone application that uses the modern OpenAI SDK, removes legacy LangChain dependencies, and adds built-in tools for semantic deduplication and terminology normalization.

## Goals

- Modernize the AutoCriteria extraction workflow.
- Replace legacy LangChain-based components with the modern OpenAI SDK.
- Provide a standalone desktop application interface.
- Support clinical trial eligibility criteria extraction from ClinicalTrials.gov records.
- Add semantic deduplication for repeated or overlapping extracted criteria.
- Add terminology normalization for mapping raw extracted terms to canonical concepts. Unmapped concepts get returned as well
- Provide configurable settings for output paths, model selection, API key management, and dictionary mappings.

## Planned Features

- ClinicalTrials.gov trial download support.
- Modern eligibility criteria extraction pipeline.
- Structured criteria output.
- Manual dictionary-based terminology normalization.
- Semantic duplicate detection.
- Cost monitoring utilities.
- Settings window for configuration.
- Analytics window for inspecting extracted results.
- Export/download support for trial data and extracted output files.
- Term Normalization and Data Download as seperate scripts

## Usage

### Backend

Once everything is setup, run `python main.py`. Logged extraction is printed, and once finished, you get a summary like:
```bash
...
PipelineSummary(trials_processed=10, criteria_extracted=311, criteria_after_normalization=311, criteria_after_deduplication=303, output_file=PosixPath('outputs/modern_autocrit_output.xlsx'), total_cost_usd=0.01949175)
```
Which should help with debugging, since it gives trials processed, how many criteria move through the pipeline, and the total cost.