# Changelog

This changelog documents the repository from its initial scaffold through the current working tree. Entries are ordered from newest to oldest and intentionally omit calendar dates.

## Unreleased

### Added

- Added the browser-based Automated Eligibility Criteria Extraction interface, ClinicalTrials.gov API ingestion, atomic eligibility segmentation, governed EVA extraction, review and reconciliation, recoverable jobs, result caching, and clinical-distance analysis.

### Changed

- Consolidated the active eligibility and registry processing packages under `backend/criteria_processor/` and `backend/ctg_parser/`.
- Replaced the former desktop and XML/Excel extraction path with the browser-first  extraction service.
- Reduced application configuration and dependencies to those used by the active browser workflow.
- Corrected the canonical attribute mapping for `ability to swallow oral medication` so it normalizes to the plural form, `ability to swallow oral medications`.
- Reconstructed this changelog from the complete Git history and organized each milestone by additions, changes, and removals.

### Removed

- Removed the unused Tkinter frontend and the `frontend_pyside_backup` application.
- Removed the superseded XML/Excel extractor, normalization pipeline, anomaly detector, integrity validator, legacy dictionary services, and their dedicated schemas and tests.
- Removed the old trial downloader, dictionary-update scripts, benchmarking placeholders, development notebook, desktop screenshots, downloaded XML cache, and generated Excel outputs.

### Next steps sketch

- Explore an embedding-based approach for adaptive dictionary maintenance, including semantic matching of unseen terms, suggested canonical mappings, confidence scoring, and a human-review step before dictionary updates are accepted.

## Validation agent and Integrity workspace

### Added

- Added a dedicated `backend.validation_agent` package with prompt construction, per-trial validation, multi-trial orchestration, structured result objects, metric calculation, report context generation, and report export.
- Added LLM-assisted comparison of source eligibility text against extracted workbook rows. The audit identifies missed or partially extracted source clauses, unsupported or potentially fabricated rows, and general quality concerns.
- Added estimated precision, recall, F1, and source-coverage metrics for integrity reviews.
- Added multi-sheet Excel integrity reports containing source findings, extraction findings, quality notes, metrics, and run status.
- Added a Tkinter **Integrity** tab with XML-directory, extraction-workbook, and report-path selectors; optional NCT ID filtering; trial-level progress; stop handling; results summaries; and follow-up chat grounded in the completed audit.
- Added validation and chat prompts, a public validation-agent package interface, an Integrity screenshot, and an integrity-validation test suite.

### Changed

- Registered the Integrity view in the desktop application's navigation and shared application state.
- Extended validation schemas to support the integrity agent's structured findings and metrics.
- Updated the README with integrity-audit behavior, limitations, controls, output details, and the fifth application tab.

## Expanded normalization dictionaries

### Changed

- Substantially expanded the active attribute, disease, and entity normalization maps with terminology observed in trial data.
- Reorganized dictionary contents into a broader deterministic mapping set while consolidating and replacing earlier entries.

## Trial downloader and dictionary updater

### Added

- Added a ClinicalTrials.gov API downloader that searches by query, paginates results, converts study records to XML, and saves individual trial files.
- Added command-line controls for query, result count, output directory, page size, request timeout, large-download confirmation, and non-interactive confirmation bypass.
- Added API and input error handling, filename sanitization, duplicate-safe downloads, and helpers for converting individual studies or dataframe rows to XML.
- Added `dict_mapping.py` to read pipeline workbooks and append previously unseen entity, attribute, and disease terms without overwriting existing mappings.
- Added reset support that restores all active dictionaries from repository-maintained originals.
- Added original baseline copies of all three dictionaries under `config/dictionaries/original/`.
- Added focused tests for downloading, XML conversion, dictionary updates, duplicate preservation, missing columns, and reset behavior.

### Changed

- Expanded the README with downloader and dictionary-maintenance commands and clarified generated output files.
- Updated dependencies required by the data-management scripts.

## Quality-control and anomaly detection

### Added

- Added structured anomaly and validation schemas for severity, category, explanation, source fields, suggested actions, and validation metrics.
- Added a deterministic anomaly detector that evaluates extracted rows and converts findings into reportable dataframes.
- Added numeric parsing and plausibility checks for measurements, comparators, ranges, dosages, laboratory values, ages, durations, scores, and related clinical values.
- Added unit-compatibility checks and terminology checks for placeholders, malformed values, suspicious attributes, and incomplete terms.
- Added pipeline generation of an anomaly workbook alongside criteria and unmapped-term outputs.
- Added anomaly counts and previews to the Analytics view, plus direct access to the generated anomaly report.
- Added application screenshots for Run, Settings, Dictionary, and Analytics workflows.
- Added package scaffolding for future consistency, unit, aggregate-metrics, and validation-agent quality modules.

### Changed

- Extended pipeline summaries and analytics state with quality-control results.
- Expanded the README into full project, feature, setup, usage, input, and output documentation.
- Updated the project notebook to exercise and inspect the quality-control workflow.

## Repository hygiene update

### Changed

- Expanded `.gitignore` coverage for environment files, Python caches, generated data, outputs, logs, editor metadata, and platform-specific files.
- Preserved tracked placeholder files where empty runtime directories are required.

## Tkinter desktop application and LLM provider abstraction

### Added

- Added a Tkinter desktop shell with Run, Settings, Dictionary, and Analytics tabs.
- Added a full Dictionary view for searching, filtering, adding, editing, deleting, and saving entity, attribute, and disease mappings.
- Added richer Analytics and Settings interfaces, including summary cards, file previews, provider/model configuration, API credentials, extraction controls, cost thresholds, directory settings, and connection testing.
- Added an abstract LLM provider interface, a normalized response model, a provider factory, and an OpenAI Responses API implementation with token accounting, retry backoff, quota-specific errors, and connection testing.
- Added configurable provider, model, base URL, temperature, and maximum-output-token settings as the foundation for OpenAI, Anthropic, Gemini, and Ollama integrations.
- Added package initializers for the new frontend layout.
- Preserved the former PySide interface and worker implementation under `frontend_pyside_backup/` for reference.

### Changed

- Replaced the active PySide GUI with Tkinter for desktop stability and simplified launch behavior.
- Refactored extraction and command-line pipeline setup from OpenAI-specific settings/client calls to the provider-neutral LLM configuration and factory.
- Expanded the Run view with threaded execution, queue-based logging, progress and status reporting, file selectors, collapsible logs, success summaries, and error handling.
- Updated routes, default model behavior, settings validation, dictionaries, dependencies, and documentation for the new application architecture.

### Removed

- Removed PySide as an active runtime dependency.
- Removed the old PySide components and worker from the active `frontend` package after retaining backup copies.
- Removed the previous OpenAI-specific settings path from the active extraction workflow.

## Background pipeline execution and logging

### Added

- Added a background pipeline worker and frontend worker package so extraction can run without freezing the interface.
- Added queue-based forwarding of Python log records into the Run view.
- Added live progress, run-state controls, output summaries, and improved failure reporting to the pipeline screen.
- Added the first repository changelog.

### Changed

- Reworked the Run view around asynchronous execution and more reliable UI logging.
- Updated the application entry point and dependencies for the revised GUI execution model.

## Benchmarking scaffold

### Added

- Added benchmark entry points for overall evaluation, model comparisons, extraction metrics, and semantic comparisons.
- Documented the planned agentic benchmarking and evaluation work.

### Removed

- Removed the original empty cost-monitor, normalizer, and parser test placeholders in favor of the benchmark scaffold.

## First graphical application

### Added

- Added the first PySide desktop application with Run, Settings, Analytics, dictionary editor, file table, and cost-panel components.
- Added application-state creation and API-style routes for settings persistence, pipeline execution, output previews, and dictionary operations.
- Added a Codex bridge abstraction for optional external task execution.
- Added a standalone `run_pipeline.py` backend entry point.
- Added a compatibility placeholder for pluralized dictionary services.

### Changed

- Simplified `main.py` to launch the desktop application.
- Enhanced normalization and attribute mapping behavior for use from the GUI.
- Updated the notebook and README to reflect the first end-to-end graphical workflow.

## Unmapped-term reporting and data utilities

### Added

- Added normalizer support for identifying and returning entity and attribute terms that do not have canonical dictionary mappings.
- Added pipeline output for unmapped terminology so mappings can be reviewed and extended.
- Added an initial ClinicalTrials.gov data-download script and a dictionary-mapping script scaffold.
- Expanded attribute and disease mappings with early real-world terminology.

### Changed

- Reduced and refocused the exploratory notebook after moving reusable behavior into project modules and scripts.
- Clarified in the README that terminology normalization returns unmapped concepts for review.

## Initial working backend

### Added

- Implemented the LLM extraction engine for chunking eligibility text, building prompts, parsing structured responses, splitting inclusion and exclusion criteria, and processing XML directories.
- Implemented criteria normalization for entities, attributes, values, exact duplicates, fuzzy duplicates, and dataframe conversion.
- Implemented robust JSON-array extraction and cleanup for model responses.
- Added detailed extraction and temporal prompt builders.
- Added Pydantic schemas for criteria, trial metadata, application settings, extraction settings, costs, directories, and dictionaries.
- Added ClinicalTrials.gov XML discovery and parsing for NCT ID, title, phase, conditions, URL, and eligibility text.
- Added OpenAI client integration with retries, JSON generation, usage tracking, and cost monitoring.
- Added the end-to-end pipeline, dictionary service, file-management helpers, logging, path management, and text-cleaning utilities.
- Added default settings, starter entity/attribute/disease maps, environment-variable documentation, a runnable backend entry point, and an exploratory notebook.

### Changed

- Expanded the README from a project stub into an overview of goals and initial usage.

## Project framework

### Added

- Added Python-oriented ignore rules for virtual environments, caches, environment secrets, generated data, logs, and output artifacts.
- Added the initial README content and dependency list.

## Repository creation

### Added

- Created the initial project hierarchy for backend API, core extraction, schemas, services, utilities, frontend components and views, configuration, dictionaries, data, documentation, logs, outputs, and tests.
- Added empty package markers, entry points, configuration placeholders, dictionary placeholders, environment example, packaging metadata, and tracked directory placeholders.
- Established the intended separation between extraction logic, desktop presentation, shared services, configuration, generated artifacts, and tests.
