# `end_to_end_pipeline_6.py` — Split-Document Classification Pipeline

**Purpose**  
This pipeline classifies Azure DI JSON inputs **document-by-document within each PDF** (not one full-file pass), then produces a final aggregated file-level classification.

---

## **What This Pipeline Does**

- Loads Azure DI JSON files (`*.pdf.json`, `*.json`)
- Detects internal document boundaries inside each PDF
- Builds document units (segments)
- Runs LLM classification per document unit
- Aggregates unit-level decisions into one final file-level label
- Exports JSON outputs for review and PDF highlighting workflows

---

## **Segmentation Rules**

1. **Continuation Rule (highest priority)**  
   If page numbering continues and page font profile is similar, keep pages in the same document unit.

2. **Dramatic Page Size Change**  
   If page dimensions change beyond threshold, start a new document unit.

3. **New Title at Page Start**  
   If a new title appears near the top and continuation is not confirmed, start a new document unit.

---

## **Key Output Fields**

Each output JSON (`*_WORKING_PageNumbers_analysis.json`) includes:

- `classification` — final aggregated file-level category
- `document_units` — per-document-unit pages, classification, findings/exceptions
- `classification_summary`  
  - `unique_categories`  
  - `concatenated_categories`  
  - `category_pages`  
  - `category_page_ranges`  
  - `category_reasons`
- `segments`, `exceptions`, `extracted_names`

---

## **Configuration**

Update these values in the script before running:

- `PROMPT_PATH`
- `INPUT_DIR`
- `OUTPUT_DIR`
- `VERSION`

Main mode flags:

- `DOCUMENT_LEVEL_ANALYSIS = True`
- `DOCUMENT_SPLIT_CLASSIFICATION = True`

Speed/coverage controls:

- `MAX_SEGMENTS_PER_FILE`
- `SEGMENT_SAMPLE_HEAD_PAGES`
- `SEGMENT_SAMPLE_TAIL_PAGES`
- `MAX_CHARS_PER_SEGMENT_CALL`

---

## **How To Run**

python3 end_to_end_pipeline_6.py
