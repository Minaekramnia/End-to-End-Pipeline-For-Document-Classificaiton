"""
END-TO-END PIPELINE FOR DOCUMENT CLASSIFICATION — Version 3
==========================================================

This pipeline processes documents for classification categories.
Simply update the CONFIGURATION section below to process different categories.

PIPELINE STEPS:
---------------
STEP 1: Initialize LangChain model with Gemini
STEP 2: Load and verify prompt file
STEP 3: Process JSON files with LLM analysis
        - Load Azure DI JSON files
        - Create document segments
        - Analyze with AI model
        - Extract names and exceptions
        - Generate analysis JSON output files
STEP 4: Highlight PDFs based on analysis results
        - Match JSON analysis files with original PDFs
        - Add highlights to PDFs using bounding boxes
        - Save highlighted PDFs

To use for a new category:
1. Update INPUT_DIR to point to your category's JSON files
2. Update OUTPUT_DIR to your desired output location
3. Update PROMPT_PATH to your prompt file
4. Update CATEGORY_NAME to your category identifier
5. Update CATEGORY_DISPLAY_NAME for logging/output
6. Update PDF_DIR to point to original PDF files
7. Update HIGHLIGHT_OUTPUT_DIR for highlighted PDFs

SPEED (Databricks): Do NOT run `%pip install langchain-google-vertexai` every run — pip resolves a huge tree
(langchain-tests, pytest, …) and can take 10–30+ minutes even when packages are already installed.
Prefer Cluster Libraries: langchain-google-vertexai, thefuzz, pymupdf, pydantic (>=2,<2.12).
"""

import os
import sys

# ============================================================================
# OPTIONAL pip — OFF by default (fast runs)
# ============================================================================
RUN_PIP_INSTALL = False
# Set True once on a bare cluster, OR export PIPELINE_RUN_PIP=1 for a single run.
# After installing, set back to False and restart the kernel.


def _optional_pip_install():
    import subprocess
    base = [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check"]
    subprocess.run(base + ["langchain-google-vertexai", "thefuzz", "PyMuPDF"], check=False)
    subprocess.run(base + ["pydantic>=2.0,<2.12"], check=False)


_env_pip = os.environ.get("PIPELINE_RUN_PIP", "").strip().lower() in ("1", "true", "yes")
if RUN_PIP_INSTALL or _env_pip:
    print("PIPELINE: installing dependencies (quiet)…")
    _optional_pip_install()
    print("PIPELINE: pip done. If versions changed, restart the Python kernel before continuing.")
else:
    print("PIPELINE: skipping pip (fast path). Set RUN_PIP_INSTALL=True or PIPELINE_RUN_PIP=1 if imports fail.")

# ============================================================================
# IMPORTS
# ============================================================================
import json
import glob
import re
from datetime import datetime

import pydantic
_pv = getattr(pydantic, "__version__", "0.0.0").split(".")
if int(_pv[0]) > 2 or (int(_pv[0]) == 2 and int(_pv[1] if len(_pv) > 1 else 0) >= 12):
    raise RuntimeError(
        f"pydantic {getattr(pydantic, '__version__', '?')} is too new for this cluster. "
        "Run the INSTALLATION cell, then Kernel > Restart Python. "
        "Need pydantic>=2.0,<2.12."
    )

from langchain_google_vertexai import ChatVertexAI
from thefuzz import fuzz

STANDARD_DOCUMENT_LEVEL_BBOX_INCHES = [0.0, 0.0, 8.5, 1.0]

# ============================================================================
# CREDENTIALS & GEMINI MODEL CONFIGURATION (WIF / GOOGLE_APPLICATION_CREDENTIALS)
# ============================================================================
# For Databricks: use the path below. For local runs, point to your JSON key path.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Workspace/Shared/Collections/config Json/itsks-ent-search-dev-proj.json"
CREDENTIALS_PATH = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
PROJECT_ID = "itsks-ent-search-dev-proj"
PROJECT_LOCATION = "global"
# NOTE: Some projects do not have access to all Gemini publisher models.
# If you see: "Publisher Model ... was not found or your project does not have access to it",
# switch this to a model you have access to (e.g. gemini-3-flash-preview).
MODEL = "gemini-3-flash-preview"
TEMPERATURE = 0.1   # Lower temperature for more stable labels and better reproducibility
MAX_TOKENS = 8192    # Hybrid single-pass sends whole doc; needs headroom for JSON (findings list)


def construct_langchain(model_name, credentials_path, temperature, max_tokens, location, project, streaming=False):
    """Build ChatVertexAI; uses GOOGLE_APPLICATION_CREDENTIALS from env. Set streaming=False for pipeline JSON parsing."""
    return ChatVertexAI(
        model_name=model_name,
        project=project,
        location=location,
        temperature=temperature,
        max_output_tokens=max_tokens,
        streaming=streaming,
    )

# ============================================================================
# CONFIGURATION — Mixed Category Hybrid Classification (edit paths only)
# ============================================================================
#
# IMPORTANT: Update paths before running. On Databricks use Volumes/Workspace paths;
# locally, point INPUT_DIR / OUTPUT_DIR to your machine.
#
# Document-level: PROMPT_PATH filename must include "document_level" → one LLM pass + top banner.
#

# Hybrid prompt (findings/classification output)
PROMPT_PATH = "/Volumes/qa_datascience_convaiqa/volumes/dataanalyticslake_convai/PRISMArchives/prompts/prompt_refactored_master_5V.md"

VERSION = "v2_6_realtest_splitdoc_final_v2"

# INPUT: Azure DI JSON files (*.pdf.json / *.json)
INPUT_DIR = "/Volumes/qa_datascience_convaiqa/volumes/dataanalyticslake_convai/PRISMArchives/RealTest/"

# Process a single document stem only, e.g. "Personal_information_9" (no .pdf / .json). Empty = all files.
# Override with env: PIPELINE_RUN_ONLY=Personal_information_9
RUN_ONLY_BASENAME = ""

# OUTPUT: analysis JSON (new folder — previous runs used .../OutputJson for comparison)
OUTPUT_DIR = "/Volumes/qa_datascience_convaiqa/volumes/dataanalyticslake_convai/PRISMArchives/Mixed/OutputJson"

# If PROMPT_PATH is missing, try these in order
PROMPT_PATH_FALLBACKS = [
    "/Users/minaekramnia/Documents/WorldBank/pdf_classifier/Azure_DI_output_parser/prompt_refactored_v2_new_version.md",
    "/Volumes/qa_datascience_convaiqa/volumes/dataanalyticslake_convai/PRISMArchives/prompts/prompt_refactored_v1.md",
]

CATEGORY_NAME = "mixed"

CATEGORY_DISPLAY_NAME = "Mixed Category Classification"

# PDF highlight output removed in this version (JSON analysis only).

# Document-level analysis (recommended): **one** Vertex/Gemini call on the **full** document text (hybrid v6).
# False = one call per segment (often ~1 segment per page → slow). Ignored when PROMPT_PATH contains "document_level"
# (that path uses the attorney single-pass branch instead).
DOCUMENT_LEVEL_ANALYSIS = True

# Segment-aware document-level mode:
# True  -> classify each separated document segment with capped text (faster on very large PDFs),
#          then aggregate to one final file-level classification.
# False -> existing behavior (one call on full file text when DOCUMENT_LEVEL_ANALYSIS=True).
DOCUMENT_SPLIT_CLASSIFICATION = True

# Cost/speed controls for DOCUMENT_SPLIT_CLASSIFICATION
MAX_SEGMENTS_PER_FILE = 140                # hard cap on classified segments per file (final balanced run)
SEGMENT_SAMPLE_HEAD_PAGES = 2              # pages sampled from start of each segment
SEGMENT_SAMPLE_TAIL_PAGES = 1              # pages sampled from end of each segment
MAX_CHARS_PER_SEGMENT_CALL = 18000         # char cap sent to model per segment

# Segmentation thresholds
DRAMATIC_PAGE_SIZE_CHANGE_THRESHOLD = 0.20  # 20%+ change indicates likely new document

# Document-level vs content-level: inferred from PROMPT_PATH (substring "document_level").
# Otherwise: per-span highlights. Use attorney_client_privilege_prompt.md for content-level.

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def create_cached_model(system_instruction, ttl="600s"):
    """Create cached model - for LangChain, we'll use system message instead."""
    print("🔄 Setting up model with system instruction...")
    print("   Note: LangChain will use system message for prompt")
    print(f"   Prompt length: {len(system_instruction)} characters")
    return system_instruction

def find_json_files(input_dir, patterns=None):
    """Find all JSON files in the input directory."""
    if patterns is None:
        patterns = ['*.pdf.json', '*.json']
    json_files = []
    for pattern in patterns:
        full_pattern = os.path.join(input_dir, pattern)
        found = glob.glob(full_pattern)
        json_files.extend(found)
    return sorted(list(set(json_files)))


def stem_json_basename(basename: str) -> str:
    """Input JSON basename -> stem (no .pdf.json / .json)."""
    if basename.endswith(".pdf.json"):
        return basename[: -len(".pdf.json")]
    if basename.endswith(".json"):
        return basename[: -len(".json")]
    return basename


def polygon_to_bbox(polygon):
    """Convert Azure DI polygon (alternating x,y, 8 floats for 4 points) to [x_min, y_min, x_max, y_max]. Units unchanged (inches for PDF)."""
    if not polygon or len(polygon) < 8:
        return None
    xs = [polygon[i] for i in range(0, len(polygon), 2)]
    ys = [polygon[i] for i in range(1, len(polygon), 2)]
    return [min(xs), min(ys), max(xs), max(ys)]


def _parse_bbox_to_floats(bbox):
    """Convert bbox to list of 4 floats. Handles list, tuple, or string (e.g. '1.2, 0.75, 8.1, 1.04' or '[1.2, 0.75, 8.1, 1.04]'). Returns None if invalid."""
    if bbox is None:
        return None
    if isinstance(bbox, (list, tuple)):
        if len(bbox) == 1 and isinstance(bbox[0], (list, tuple)):
            bbox = list(bbox[0])
        else:
            bbox = list(bbox)
    elif isinstance(bbox, str):
        s = bbox.strip().strip("[]")
        try:
            bbox = [float(x.strip()) for x in s.split(",") if x.strip()]
        except (ValueError, AttributeError):
            return None
    else:
        return None
    if len(bbox) == 8:
        try:
            xs = [float(bbox[i]) for i in range(0, 8, 2)]
            ys = [float(bbox[i]) for i in range(1, 8, 2)]
            return [min(xs), min(ys), max(xs), max(ys)]
        except (ValueError, TypeError, IndexError):
            return None
    if len(bbox) == 4:
        try:
            return [float(x) for x in bbox]
        except (ValueError, TypeError):
            return None
    return None


def normalize_exception_bbox(exception):
    """Normalize exception so bounding_box is [x1,y1,x2,y2] in inches. Returns new dict or None if invalid.
    Handles bounding_box, box_2d, 4- or 8-element, nested list, or string; if values > 50 treats as PDF points (÷72)."""
    bbox = exception.get("bounding_box") or exception.get("box_2d")
    parsed = _parse_bbox_to_floats(bbox)
    if parsed is None:
        return None
    x1, y1, x2, y2 = parsed
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if max(x1, y1, x2, y2) > 50:
        x1, y1, x2, y2 = x1 / 72.0, y1 / 72.0, x2 / 72.0, y2 / 72.0
    out = dict(exception)
    out["bounding_box"] = [x1, y1, x2, y2]
    return out


def _response_content_to_str(content):
    """Convert LangChain response.content to str; Gemini/LangChain may return a list of content parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            else:
                parts.append(str(p))
        return "".join(parts).strip()
    return str(content).strip()


def _bbox_from_segment_content(segment_content, page_number, pages_in_segment):
    """Extract first [Page N, bbox: x,y,x,y] from segment_content for the given page (or first page in segment). Returns [x1,y1,x2,y2] or None."""
    target_page = page_number
    if target_page is None and pages_in_segment:
        target_page = int(pages_in_segment[0])
    if target_page is None:
        target_page = 1
    try:
        target_page = int(target_page)
    except (TypeError, ValueError):
        target_page = 1
    # Match [Page N, bbox: 1.2, 3.4, 5.6, 7.8] or [Page N, bbox: 1.2,3.4,5.6,7.8]
    pattern = rf"\[Page\s+{target_page}\s*,\s*bbox:\s*([\d.,\s\-]+)\]"
    m = re.search(pattern, segment_content)
    if m:
        try:
            bbox = [float(x.strip()) for x in m.group(1).split(",") if x.strip()][:4]
            if len(bbox) == 4:
                return bbox
        except (ValueError, AttributeError):
            pass
    # Fallback: any [Page *, bbox: ...] in segment (use first match for that page or first in segment)
    for p in ([target_page] + [p for p in (pages_in_segment or []) if p != target_page]):
        pattern = rf"\[Page\s+{p}\s*,\s*bbox:\s*([\d.,\s\-]+)\]"
        m = re.search(pattern, segment_content)
        if m:
            try:
                bbox = [float(x.strip()) for x in m.group(1).split(",") if x.strip()][:4]
                if len(bbox) == 4:
                    return bbox
            except (ValueError, AttributeError):
                pass
    # Default so we still get a highlight (e.g. top-left region of page in inches)
    return [0.5, 0.5, 8.0, 2.0]


# ============================================================================
# DOCUMENT PROCESSOR CLASS
# ============================================================================
class DocumentProcessor:
    """Document processor for Azure DI JSON files."""
    
    def __init__(self, filepath, langchain_model=None, system_instruction=None, document_level_mode=False, document_level_analysis=False):
        self.filepath = filepath
        self.analyze_result = None
        self.page_content = {}
        self.page_dimensions = {}
        self.page_font_signatures = {}
        self.document_segments = {}
        self.langchain_model = langchain_model
        self.system_instruction = system_instruction
        self.document_level_mode = document_level_mode
        self.document_level_analysis = document_level_analysis
        self._load_data()
    
    def _load_data(self):
        """Load JSON file."""
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.analyze_result = data.get('analyzeResult')
            if not self.analyze_result:
                raise ValueError("Missing 'analyzeResult' key")
            print(f"✅ Loaded: {os.path.basename(self.filepath)}")
        except Exception as e:
            print(f"❌ Error: {e}")
            self.analyze_result = None
    
    def _organize_content_by_page(self):
        """Organize content by page."""
        if not self.analyze_result:
            return
        
        # Initialize pages
        for page in self.analyze_result.get('pages', []):
            page_number = page.get('pageNumber')
            if page_number:
                self.page_content[page_number] = {'paragraphs': [], 'tables': [], 'page_numbers': []}
                self.page_dimensions[page_number] = {
                    'width': page.get('width'),
                    'height': page.get('height'),
                    'unit': page.get('unit')
                }
                self.page_font_signatures[page_number] = self._compute_page_font_signature(page)
        
        # Process paragraphs (Azure DI can use 'content' directly or 'spans' into top-level content)
        top_content = self.analyze_result.get('content') or ''
        page_numbers_list = sorted(self.page_content.keys()) if self.page_content else [1]
        for paragraph in self.analyze_result.get('paragraphs', []):
            regions = paragraph.get('boundingRegions') or []
            if not regions:
                regions = [{}]
            page_number = regions[0].get('pageNumber') if (regions and isinstance(regions[0], dict)) else None
            if page_number not in self.page_content:
                page_number = page_numbers_list[0] if page_numbers_list else 1
            if page_number not in self.page_content:
                continue
            # Get content: direct 'content' or extract from top-level via 'spans' (Azure DI v4)
            para_content = paragraph.get('content') or ''
            if not para_content and paragraph.get('spans') and top_content:
                for span in paragraph.get('spans', []):
                    off = span.get('offset', 0)
                    length = span.get('length', 0)
                    if isinstance(off, int) and isinstance(length, int):
                        try:
                            para_content += top_content[off:off + length]
                        except (IndexError, TypeError):
                            pass
            polygon = regions[0].get('polygon', []) if isinstance(regions[0], dict) else []
            para_info = {
                'content': para_content.strip() if isinstance(para_content, str) else '',
                'role': paragraph.get('role', 'paragraph'),
                'boundingBox': polygon
            }
            if para_info['role'] == 'pageNumber':
                self.page_content[page_number]['page_numbers'].append(para_info)
            else:
                self.page_content[page_number]['paragraphs'].append(para_info)
        
        # Fallback: if no paragraph content on any page, use top-level content (Azure DI may expose only content + spans)
        total_para_content = sum(
            len(p.get('content', '')) for pg in self.page_content.values() for p in pg.get('paragraphs', [])
        )
        if total_para_content == 0 and top_content.strip():
            first_page = min(self.page_content.keys()) if self.page_content else 1
            self.page_content.setdefault(first_page, {'paragraphs': [], 'tables': [], 'page_numbers': []})
            self.page_content[first_page]['paragraphs'].append({
                'content': top_content.strip(),
                'role': 'paragraph',
                'boundingBox': []
            })
            print(f"   ⚠️  No paragraph content from structure; using top-level content on page {first_page} ({len(top_content)} chars)")
        
        # Process tables
        for table in self.analyze_result.get('tables', []):
            if table.get('cells') and table['cells'][0].get('boundingRegions'):
                page_number = table['cells'][0]['boundingRegions'][0].get('pageNumber')
                if page_number in self.page_content:
                    self.page_content[page_number]['tables'].append(table)
    
    def _parse_page_number(self, content):
        """Parse page number content to extract sequence info."""
        content = content.strip().lower()
        
        if not re.match(r'^[\d\s/]+$', content):
            return None
        
        match = re.match(r'^(\d+)/(\d+)$', content)
        if match:
            return (int(match.group(1)), int(match.group(2)), content)
        
        match = re.match(r'^page\s*(\d+)$', content)
        if match:
            return (int(match.group(1)), None, content)
        
        match = re.match(r'^(\d+)$', content)
        if match:
            return (int(match.group(1)), None, content)
        
        return None

    @staticmethod
    def _line_height_from_polygon(polygon):
        """Approximate text line height from Azure polygon coordinates."""
        if not polygon or len(polygon) < 8:
            return None
        ys = []
        try:
            ys = [float(polygon[i]) for i in range(1, len(polygon), 2)]
        except (TypeError, ValueError):
            return None
        if not ys:
            return None
        h = max(ys) - min(ys)
        return h if h > 0 else None

    def _compute_page_font_signature(self, page_obj):
        """Compute lightweight page 'font profile' using line-height stats."""
        lines = page_obj.get("lines", []) or []
        heights = []
        for ln in lines[:400]:
            h = self._line_height_from_polygon(ln.get("polygon", []))
            if h is not None:
                heights.append(h)
        if not heights:
            return None
        heights.sort()
        n = len(heights)
        median_h = heights[n // 2]
        avg_h = sum(heights) / n
        return {"median_h": median_h, "avg_h": avg_h, "n": n}

    def _is_font_profile_similar(self, prev_page_num, curr_page_num):
        """True when two pages have similar line-height profile."""
        prev_sig = self.page_font_signatures.get(prev_page_num)
        curr_sig = self.page_font_signatures.get(curr_page_num)
        if not prev_sig or not curr_sig:
            # Missing style info: do not block continuation.
            return True
        prev_med = float(prev_sig.get("median_h", 0) or 0)
        curr_med = float(curr_sig.get("median_h", 0) or 0)
        if prev_med <= 0 or curr_med <= 0:
            return True
        rel_change = abs(prev_med - curr_med) / max(prev_med, 1e-6)
        return rel_change <= 0.25
    
    def _get_page_number_sequence(self, page_num):
        """Extract page number sequence information from a page."""
        page_data = self.page_content.get(page_num, {})
        
        page_numbers = page_data.get('page_numbers', [])
        if page_numbers:
            content = page_numbers[0].get('content', '').strip()
            match = self._parse_page_number(content)
            if match:
                return match
        
        for para in page_data.get('paragraphs', []):
            content = para.get('content', '').strip()
            match = self._parse_page_number(content)
            if match:
                return match
        
        return None
    
    def _is_continuation_of_previous_page(self, page_num):
        """Check if this page continues a page number sequence from previous page."""
        if page_num not in self.page_content:
            return False
        
        sorted_pages = sorted(self.page_content.keys())
        page_index = sorted_pages.index(page_num)
        
        if page_index == 0:
            return False
        
        prev_page = sorted_pages[page_index - 1]
        current_seq = self._get_page_number_sequence(page_num)
        prev_seq = self._get_page_number_sequence(prev_page)
        
        if not current_seq or not prev_seq:
            return False
        
        current_num, current_total, current_text = current_seq
        prev_num, prev_total, prev_text = prev_seq
        
        if current_total and prev_total:
            if current_total == prev_total and current_num == prev_num + 1:
                return True
        elif current_num and prev_num and not current_total and not prev_total:
            if current_num == prev_num + 1:
                return True
        
        return False

    def _is_new_document_boundary(self, page_num):
        """Boundary detection tuned for archive scans.
        Rule 1: page number continuation + similar font => same document.
        Rule 2: dramatic page size change => new document.
        Rule 3: title at page start => new document (when not continued).
        """
        if page_num not in self.page_content:
            return False

        sorted_pages = sorted(self.page_content.keys())
        page_index = sorted_pages.index(page_num)
        if page_index == 0:
            return True  # first page always starts a segment

        prev_page = sorted_pages[page_index - 1]
        page_data = self.page_content.get(page_num, {})

        has_title = any(p.get("role") == "title" for p in page_data.get("paragraphs", [])[:3])

        prev_dims = self.page_dimensions.get(prev_page, {})
        curr_dims = self.page_dimensions.get(page_num, {})
        prev_h = float(prev_dims.get("height", 0) or 0)
        curr_h = float(curr_dims.get("height", 0) or 0)
        prev_w = float(prev_dims.get("width", 0) or 0)
        curr_w = float(curr_dims.get("width", 0) or 0)
        height_change = abs(prev_h - curr_h) / max(prev_h, 1.0)
        width_change = abs(prev_w - curr_w) / max(prev_w, 1.0)
        has_dramatic_size_change = (
            (height_change > DRAMATIC_PAGE_SIZE_CHANGE_THRESHOLD)
            or (width_change > DRAMATIC_PAGE_SIZE_CHANGE_THRESHOLD)
        )

        is_continuation = self._is_continuation_of_previous_page(page_num)
        same_font_profile = self._is_font_profile_similar(prev_page, page_num)

        # New order for lower over-segmentation:
        # if sequence continues and page style is similar, keep same document.
        if is_continuation and same_font_profile:
            return False
        if is_continuation and not same_font_profile:
            print(f"   🔤 Page {page_num}: page-number continuation found, but font profile changed")
        if has_dramatic_size_change:
            print(f"   📏 Page {page_num}: NEW document (dramatic size change)")
            return True
        if has_title:
            print(f"   📄 Page {page_num}: NEW document (new title)")
            return True
        return False
    
    def _is_record_removal_notice_page(self, page_num):
        """Check if a page is a Record Removal Notice page."""
        if page_num not in self.page_content:
            return False
        
        page_data = self.page_content.get(page_num, {})
        target_string = "Record Removal Notice"
        # Strong signal: actual words (avoids WRatio false positives on numeric/table pages).
        paragraphs = page_data.get('paragraphs', [])
        blob = " ".join((p.get("content") or "") for p in paragraphs[:40]).lower()
        if "record removal" in blob:
            return True

        # OCR / layout typos on the title line only (short lines, fuzzy).
        THRESHOLD = 88
        for para in paragraphs[:5]:
            content = (para.get("content") or "").strip()
            if not content or len(content) > 120:
                continue
            if fuzz.WRatio(content, target_string) >= THRESHOLD:
                return True

        return False
    
    def _filter_record_removal_notice_pages(self):
        """Remove Record Removal Notice pages from processing."""
        if not self.page_content:
            return
        
        pages_to_remove = []
        for page_num in sorted(self.page_content.keys()):
            if self._is_record_removal_notice_page(page_num):
                pages_to_remove.append(page_num)

        if not pages_to_remove:
            return
        # Never drop every page — fuzzy match can flag whole short scans; empty pages => no LLM input.
        if len(pages_to_remove) >= len(self.page_content):
            print(
                f"   ⚠️  {len(pages_to_remove)} page(s) look like Record Removal Notice, but that would remove "
                f"all {len(self.page_content)} page(s). Keeping all pages for this run."
            )
            return

        for page_num in pages_to_remove:
            del self.page_content[page_num]
            if page_num in self.page_dimensions:
                del self.page_dimensions[page_num]

        print(f"✅ Excluded {len(pages_to_remove)} Record Removal Notice page(s)")
    
    def _create_segments(self):
        """Create document segments for processing."""
        self._organize_content_by_page()
        self._filter_record_removal_notice_pages()
        
        if not self.page_content:
            return
        
        segments = []
        current_segment = {'pages': [], 'content': []}
        
        for page_num in sorted(self.page_content.keys()):
            page_data = self.page_content[page_num]
            
            # Collect all text from page (include bbox so LLM can return it for highlights)
            page_text_parts = []
            
            # Add paragraphs with bbox prefix (Azure DI polygon -> [x_min,y_min,x_max,y_max] in inches)
            # If no polygon, use default bbox so the model always has numbers to copy (avoids dropped exceptions)
            DEFAULT_BBOX = [0.5, 0.5, 8.0, 2.0]
            for para in page_data.get('paragraphs', []):
                content = para.get('content', '').strip()
                if not content:
                    continue
                bbox = polygon_to_bbox(para.get('boundingBox', []))
                if not bbox:
                    bbox = DEFAULT_BBOX
                bbox_str = ",".join(map(str, bbox))
                page_text_parts.append(f"[Page {page_num}, bbox: {bbox_str}]\n{content}")
            
            # Add tables (no per-cell bbox for now)
            for table in page_data.get('tables', []):
                table_text = self._format_table(table)
                if table_text:
                    page_text_parts.append(f"[Page {page_num}]\n{table_text}")
            
            page_text = "\n\n".join(page_text_parts)
            
            # 3-rule boundary detection (legacy behavior recovery)
            is_new_document = self._is_new_document_boundary(page_num)
            if not is_new_document:
                current_segment['pages'].append(page_num)
                current_segment['content'].append(f"Page {page_num}:\n{page_text}")
            else:
                # Save previous segment if it exists
                if current_segment['pages']:
                    segments.append(current_segment)
                
                # Start new segment
                current_segment = {
                    'pages': [page_num],
                    'content': [f"Page {page_num}:\n{page_text}"]
                }
        
        # Add final segment
        if current_segment['pages']:
            segments.append(current_segment)
        
        self.document_segments = {i: seg for i, seg in enumerate(segments, 1)}
        for sid, seg in self.document_segments.items():
            seg_len = sum(len(c) for c in seg.get('content', []))
            print(f"   Segment {sid}: {len(seg.get('content', []))} block(s), {seg_len} chars")
        print(f"✅ Created {len(segments)} segment(s)")
    
    def _format_table(self, table):
        """Format table as text."""
        if not table.get('cells'):
            return ""
        
        rows = {}
        for cell in table['cells']:
            row_index = cell.get('rowIndex', 0)
            col_index = cell.get('columnIndex', 0)
            content = cell.get('content', '').strip()
            
            if row_index not in rows:
                rows[row_index] = {}
            rows[row_index][col_index] = content
        
        if not rows:
            return ""
        
        # Format as simple text table
        lines = []
        for row_idx in sorted(rows.keys()):
            row_data = rows[row_idx]
            cols = [row_data.get(col_idx, '') for col_idx in sorted(row_data.keys())]
            lines.append(" | ".join(cols))
        
        return "\n".join(lines)
    
    def _get_full_document_text(self):
        """Concatenate all segment text for one-shot document-level classification."""
        parts = []
        for seg_id in sorted(self.document_segments.keys()):
            seg = self.document_segments[seg_id]
            parts.append("\n\n".join(seg.get("content", [])))
        text = "\n\n".join(parts).strip()
        if not text and self.analyze_result:
            text = (self.analyze_result.get("content") or "").strip()
        return text

    @staticmethod
    def _is_general_label(category):
        c = (category or "").strip().lower()
        if not c:
            return True
        return ("general" in c) or ("no sensitive" in c)

    @staticmethod
    def _compress_pages(pages):
        """Convert sorted page list to compact ranges like '1-3, 6, 8-9'."""
        vals = sorted({int(p) for p in (pages or []) if isinstance(p, (int, float)) or str(p).isdigit()})
        if not vals:
            return ""
        out = []
        start = prev = vals[0]
        for p in vals[1:]:
            if p == prev + 1:
                prev = p
                continue
            out.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = prev = p
        out.append(f"{start}-{prev}" if start != prev else f"{start}")
        return ", ".join(out)

    def _build_segment_compact_text(self, segment):
        """Build compact segment text sample to avoid very large token inputs."""
        content_blocks = segment.get("content", []) or []
        pages = segment.get("pages", []) or []
        if not content_blocks:
            return ""
        n = len(content_blocks)
        take_idx = []
        for i in range(min(SEGMENT_SAMPLE_HEAD_PAGES, n)):
            take_idx.append(i)
        for i in range(max(0, n - SEGMENT_SAMPLE_TAIL_PAGES), n):
            take_idx.append(i)
        # unique while preserving order
        seen = set()
        unique_idx = []
        for i in take_idx:
            if i not in seen:
                seen.add(i)
                unique_idx.append(i)
        sampled_parts = []
        for i in unique_idx:
            p = pages[i] if i < len(pages) else "?"
            sampled_parts.append(f"[Sampled page {p}]\n{content_blocks[i]}")
        text = "\n\n".join(sampled_parts).strip()
        if len(text) > MAX_CHARS_PER_SEGMENT_CALL:
            text = text[:MAX_CHARS_PER_SEGMENT_CALL]
        return text

    def _aggregate_segment_results(self, units):
        """Aggregate segment-level classifications to a single file-level classification."""
        candidates = []
        for u in units:
            cls = u.get("classification") or {}
            category = cls.get("category")
            conf = cls.get("confidence_score")
            try:
                conf = float(conf) if conf is not None else 0.0
            except (TypeError, ValueError):
                conf = 0.0
            if category:
                candidates.append((self._is_general_label(category), conf, category, cls))
        if not candidates:
            return {"category": "General / No Sensitive Category", "confidence_score": 0.5, "reason": "No segment classification returned."}
        # Prefer non-general labels first, then confidence
        candidates.sort(key=lambda x: (x[0], -x[1]))
        best = candidates[0][3]
        # Lightweight reason if model omitted one
        if not best.get("reason"):
            best = dict(best)
            best["reason"] = "Aggregated from separated document segments."
        return best

    def _build_classification_summary(self, units):
        """Build highlight-friendly summary: unique categories + pages + reasons."""
        category_pages = {}
        category_reasons = {}
        ordered_unique = []
        for u in units:
            cls = u.get("classification") or {}
            cat = (cls.get("category") or "").strip()
            if not cat or self._is_general_label(cat):
                continue
            if cat not in category_pages:
                category_pages[cat] = set()
            for p in (u.get("pages") or []):
                try:
                    category_pages[cat].add(int(p))
                except (TypeError, ValueError):
                    pass
            reason = (cls.get("reason") or "").strip()
            if reason and cat not in category_reasons:
                category_reasons[cat] = reason
            if cat not in ordered_unique:
                ordered_unique.append(cat)

        non_sensitive = "General / No Sensitive Category"
        if not ordered_unique:
            ordered_unique = [non_sensitive]

        category_pages_list = {
            cat: sorted(list(pages))
            for cat, pages in category_pages.items()
        }
        category_page_ranges = {
            cat: self._compress_pages(pages)
            for cat, pages in category_pages.items()
        }
        concatenated = " | ".join(ordered_unique)

        return {
            "unique_categories": ordered_unique,
            "concatenated_categories": concatenated,
            "category_pages": category_pages_list,
            "category_page_ranges": category_page_ranges,
            "category_reasons": category_reasons,
        }

    def _process_full_document_with_llm(self):
        """Single LLM call for document-level prompt (v3.5-style JSON: document_level + exceptions)."""
        from langchain_core.messages import HumanMessage, SystemMessage
        content = self._get_full_document_text()
        if not content:
            return None
        format_reminder = """

REMINDER: Return ONLY valid JSON. No other text.
Follow the system prompt output structure exactly (document_level and/or exceptions with standardized bbox for document-level).
"""
        user_content = f"Full document content to analyze:\n\n{content}{format_reminder}"
        system_msg = SystemMessage(content=self.system_instruction)
        human_msg = HumanMessage(content=user_content)
        response = self.langchain_model.invoke([system_msg, human_msg])
        response_text = _response_content_to_str(response.content)
        text_stripped = response_text.strip()
        if text_stripped.startswith("```"):
            text_stripped = re.sub(r'^```(?:json)?\s*', '', text_stripped)
            text_stripped = re.sub(r'\s*```\s*$', '', text_stripped)
        result = None
        for candidate in [text_stripped, response_text]:
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    result = parsed
                    break
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    result = parsed[0]
                    break
            except (json.JSONDecodeError, TypeError):
                continue
        if result is None:
            m = re.search(r'\{.*\}', response_text, re.DOTALL)
            if m:
                try:
                    result = json.loads(m.group(0))
                except (json.JSONDecodeError, TypeError):
                    result = None
        if not isinstance(result, dict):
            return None
        # Normalize: merge document_level into v3.5-style exceptions for downstream + PDF banner
        dl = result.get("document_level") or {}
        ex_list = list(result.get("exceptions") or [])
        if dl.get("contains_attorney_client_privilege") is True and not ex_list:
            ex_list.append({
                "category": "2.4 Attorney-Client Privilege",
                "text": (content[:500] + "…") if len(content) > 500 else content,
                "bounding_box": list(STANDARD_DOCUMENT_LEVEL_BBOX_INCHES),
                "page_number": 1,
                "confidence_score": float(dl.get("confidence_score", 0.85)),
                "reason": (dl.get("reason") or "Document contains Attorney-Client Privilege content.")
                + " Document-level classification - entire document is 2.4 Attorney-Client Privilege.",
            })
        elif ex_list:
            for i, e in enumerate(ex_list):
                if isinstance(e, dict) and (dl.get("contains_attorney_client_privilege") is True or "Attorney-Client" in (e.get("category") or "")):
                    e = dict(e)
                    e.setdefault("bounding_box", list(STANDARD_DOCUMENT_LEVEL_BBOX_INCHES))
                    e.setdefault("page_number", 1)
                    ex_list[i] = e
        result["exceptions"] = ex_list
        result.setdefault("extracted_names", [])
        if "document_level" not in result and dl:
            result["document_level"] = dl
        return result

    def process_document(self, output_filepath=None):
        """Process document and generate analysis."""
        if not self.analyze_result:
            print("❌ No data loaded")
            return None
        
        print(f"\n📄 Processing document: {os.path.basename(self.filepath)}")
        
        # Create segments
        self._create_segments()
        
        if not self.document_segments:
            print("⚠️  No segments created")
            return None

        # Document-level: one LLM call on full text (matches attorney_client_privilege_prompt_document_level + future v3.5 merge)
        if self.document_level_mode:
            print("   Mode: document-level (single pass on full document)")
            result = self._process_full_document_with_llm()
            if not result:
                print("⚠️  Document-level LLM returned no result")
                return None
            unique_names = list(dict.fromkeys(result.get("extracted_names") or []))
            all_exceptions = result.get("exceptions") or []
            page_list = sorted(self.page_content.keys()) if self.page_content else [1]
            output = {
                "extracted_names": unique_names,
                "document_level": result.get("document_level") or {},
                "segments": [{"segment_id": 1, "pages": page_list, "exceptions": all_exceptions}],
                "exceptions": all_exceptions,
            }
            if output_filepath:
                os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
                print(f"\n✅ Saved to: {output_filepath} (document-level, {len(all_exceptions)} exception(s))")
            return output

        # Hybrid v6: document-level — one LLM call on full document text
        if self.document_level_analysis:
            if DOCUMENT_SPLIT_CLASSIFICATION:
                print("   Mode: document-level split classification (per separated segment, capped input)")
                all_extracted_names = []
                all_exceptions = []
                segment_units = []
                seg_items = sorted(self.document_segments.items(), key=lambda x: int(x[0]))
                if len(seg_items) > MAX_SEGMENTS_PER_FILE:
                    print(f"   ⚠️  Segment count {len(seg_items)} exceeds cap {MAX_SEGMENTS_PER_FILE}; truncating for speed.")
                    seg_items = seg_items[:MAX_SEGMENTS_PER_FILE]
                for seg_id, segment in seg_items:
                    pages_in_segment = segment.get("pages", [])
                    compact = self._build_segment_compact_text(segment)
                    if not compact:
                        continue
                    print(f"\n   Processing segment {seg_id}/{len(seg_items)} (pages {pages_in_segment[:1]}..{pages_in_segment[-1:]}, {len(compact)} chars)")
                    result = self._process_segment_with_llm(compact, seg_id, pages_in_segment, document_level_input=False)
                    if not result:
                        continue
                    all_extracted_names.extend(result.get("extracted_names") or [])
                    seg_exceptions = []
                    for e in result.get("exceptions") or []:
                        normalized = normalize_exception_bbox(e)
                        if normalized is not None:
                            seg_exceptions.append(normalized)
                            all_exceptions.append(normalized)
                            continue
                        fb = _bbox_from_segment_content(compact, e.get("page_number"), pages_in_segment)
                        if fb is not None:
                            recovered = dict(e)
                            recovered["bounding_box"] = fb
                            recovered.setdefault("page_number", pages_in_segment[0] if pages_in_segment else 1)
                            seg_exceptions.append(recovered)
                            all_exceptions.append(recovered)
                    segment_units.append({
                        "segment_id": seg_id,
                        "pages": pages_in_segment,
                        "input_chars": len(compact),
                        "classification": result.get("classification"),
                        "findings": result.get("findings"),
                        "exceptions": seg_exceptions,
                    })
                if not segment_units:
                    print("⚠️  No segment-level model outputs returned")
                    return None
                unique_names = list(dict.fromkeys(all_extracted_names))
                final_classification = self._aggregate_segment_results(segment_units)
                classification_summary = self._build_classification_summary(segment_units)
                output = {
                    "analysis_scope": "document_level_split_by_segment",
                    "extracted_names": unique_names,
                    "classification": final_classification,
                    "classification_summary": classification_summary,
                    "document_units": segment_units,
                    "segments": [
                        {
                            "segment_id": u["segment_id"],
                            "pages": u["pages"],
                            "exceptions": u.get("exceptions", []),
                        } for u in segment_units
                    ],
                    "exceptions": all_exceptions,
                }
                if output_filepath:
                    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        json.dump(output, f, indent=2, ensure_ascii=False)
                    print(f"\n✅ Saved to: {output_filepath} (split document-level, {len(segment_units)} unit(s), {len(all_exceptions)} finding(s)/exception(s))")
                return output
            else:
                print("   Mode: document-level analysis (one LLM call on full document)")
                full_content = self._get_full_document_text()
                if not full_content:
                    print("⚠️  Empty document text")
                    return None
                page_list = sorted(self.page_content.keys()) if self.page_content else [1]
                result = self._process_segment_with_llm(full_content, 1, page_list, document_level_input=True)
                if not result:
                    print("⚠️  Document-level LLM returned no result")
                    return None
                all_extracted_names = list(result.get("extracted_names") or [])
                all_exceptions = []
                for e in result.get("exceptions") or []:
                    normalized = normalize_exception_bbox(e)
                    if normalized is not None:
                        all_exceptions.append(normalized)
                        continue
                    fb = _bbox_from_segment_content(full_content, e.get("page_number"), page_list)
                    if fb is not None:
                        recovered = dict(e)
                        recovered["bounding_box"] = fb
                        recovered.setdefault("page_number", page_list[0])
                        all_exceptions.append(recovered)
                unique_names = list(dict.fromkeys(all_extracted_names))
                output = {
                    "analysis_scope": "document_level",
                    "extracted_names": unique_names,
                    "classification": result.get("classification"),
                    "findings": result.get("findings"),
                    "segments": [{"segment_id": 1, "pages": page_list, "exceptions": all_exceptions}],
                    "exceptions": all_exceptions,
                }
                if output_filepath:
                    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        json.dump(output, f, indent=2, ensure_ascii=False)
                    print(f"\n✅ Saved to: {output_filepath} (document-level, {len(all_exceptions)} finding(s)/exception(s))")
                return output
        
        # Process each segment
        all_extracted_names = []
        all_exceptions = []
        
        for seg_id, segment in self.document_segments.items():
            print(f"\n   Processing segment {seg_id}/{len(self.document_segments)}")
            
            segment_content = "\n\n".join(segment['content'])
            pages_in_segment = segment['pages']
            
            # Process with LLM
            result = self._process_segment_with_llm(segment_content, seg_id, pages_in_segment)
            
            if result:
                # Extract names
                if 'extracted_names' in result:
                    all_extracted_names.extend(result['extracted_names'])
                
                # Extract exceptions and normalize bbox to 4-element inches (so highlighting works)
                if 'exceptions' in result:
                    for e in result['exceptions']:
                        normalized = normalize_exception_bbox(e)
                        if normalized is not None:
                            all_exceptions.append(normalized)
                        else:
                            # Fallback: LLM may have returned valid exception but bbox missing/wrong format — recover bbox from segment so we still flag and highlight
                            fallback_bbox = _bbox_from_segment_content(segment_content, e.get("page_number"), pages_in_segment)
                            if fallback_bbox is not None:
                                first_page = int(pages_in_segment[0]) if pages_in_segment else 1
                                page_num = e.get("page_number")
                                if page_num is not None:
                                    try:
                                        first_page = int(page_num)
                                    except (TypeError, ValueError):
                                        pass
                                recovered = dict(e)
                                recovered["bounding_box"] = fallback_bbox
                                recovered["page_number"] = first_page
                                all_exceptions.append(recovered)
                                print(f"   ✅ Recovered 1 exception with bbox from segment (page {first_page}) so it can be highlighted.")
                            else:
                                print(f"   ⚠️  Dropped 1 exception (invalid bbox): page_number={e.get('page_number')}, bbox={e.get('bounding_box') or e.get('box_2d')}")
            
        # Remove duplicate names
        unique_names = list(dict.fromkeys(all_extracted_names))
        
        # Create final output
        output = {
            "extracted_names": unique_names,
            "segments": [
                {
                    "segment_id": seg_id,
                    "pages": seg['pages'],
                    "exceptions": [e for e in all_exceptions if any(p in seg['pages'] for p in [e.get('page_number', 0)])]
                }
                for seg_id, seg in self.document_segments.items()
            ],
            "exceptions": all_exceptions
        }
        
        # Save output
        if output_filepath:
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            total_exceptions = len(output.get('exceptions', []))
            print(f"\n✅ Saved to: {output_filepath} (total exceptions: {total_exceptions})")
        
        return output
    
    def _process_segment_with_llm(self, content, segment_id, pages_in_segment, document_level_input=False):
        """Process segment with LLM. Appends format reminder (like other exception pipelines)."""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            
            format_reminder = """

REMINDER: Return ONLY valid JSON. No other text.
- Follow the prompt schema exactly.
- If output uses "findings", include page_number and bounding_box [x1,y1,x2,y2] for each finding, copied from the input bbox lines.
- If output uses "classification", return only the classification object plus extracted_names.
"""
            prefix = (
                "Full document (all pages) to analyze:\n\n"
                if document_level_input
                else "Document Content to Analyze:\n\n"
            )
            user_content = f"{prefix}{content}{format_reminder}"
            # Debug: print what we send for segment 1 so we can verify bbox lines and content
            if segment_id == 1:
                preview_len = min(2000, len(content))
                dbg = "Document-level" if document_level_input else "Segment 1"
                print(f"   [DEBUG {dbg} input preview, first {preview_len} chars]:\n{content[:preview_len]}")
                if len(content) > preview_len:
                    print(f"   ... ({len(content) - preview_len} more chars)")
            system_msg = SystemMessage(content=self.system_instruction)
            human_msg = HumanMessage(content=user_content)
            
            response = self.langchain_model.invoke([system_msg, human_msg])
            response_text = _response_content_to_str(response.content)
            
            # Debug: for first segment, log start of raw response
            if segment_id == 1 and response_text:
                preview = response_text[:500] if len(response_text) > 500 else response_text
                print(f"   [Segment 1 raw response preview]: {preview!r}")
            
            # Parse LLM response: may be raw JSON object, list of one object, or markdown-wrapped
            result = None
            text_stripped = response_text.strip()
            # Remove markdown code fence if present
            if text_stripped.startswith("```"):
                text_stripped = re.sub(r'^```(?:json)?\s*', '', text_stripped)
                text_stripped = re.sub(r'\s*```\s*$', '', text_stripped)
            for json_str_candidate in [text_stripped, response_text]:
                if not json_str_candidate:
                    continue
                try:
                    parsed = json.loads(json_str_candidate)
                    if isinstance(parsed, dict):
                        result = parsed
                        break
                    if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                        result = parsed[0]
                        break
                except (json.JSONDecodeError, TypeError):
                    continue
                if result is not None:
                    break
            if result is None:
                m = re.search(r'\{.*\}', response_text, re.DOTALL)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                        result = parsed if isinstance(parsed, dict) else (parsed[0] if isinstance(parsed, list) and parsed else None)
                    except (json.JSONDecodeError, TypeError, IndexError):
                        pass
            if result is not None:
                # Normalize hybrid schema (findings/classification) to internal pipeline shape.
                if isinstance(result.get('findings'), list):
                    result['exceptions'] = result.get('findings', [])
                if not isinstance(result.get('exceptions'), list):
                    result['exceptions'] = []
                # Ensure we have dict with list fields (some models return wrong shape)
                if not isinstance(result.get('extracted_names'), list):
                    result['extracted_names'] = []
                n_names = len(result['extracted_names'])
                n_exc = len(result['exceptions'])
                if document_level_input:
                    print(f"   ✅ Document-level: {n_names} names, {n_exc} findings/exceptions")
                else:
                    print(f"   ✅ Segment {segment_id}: {n_names} names, {n_exc} exceptions")
                return result
            else:
                print(f"⚠️  No JSON found in response for segment {segment_id}")
                if segment_id == 1:
                    print(f"   Raw response (first 300 chars): {response_text[:300]!r}")
                return None
                
        except Exception as e:
            print(f"❌ Error processing segment {segment_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("="*80)
print(f"🚀 {CATEGORY_DISPLAY_NAME.upper()} PROCESSING")
print("="*80)

# Resolve prompt path: try PROMPT_PATH first, then fallbacks
_resolved_prompt_path = None
if os.path.exists(PROMPT_PATH):
    _resolved_prompt_path = PROMPT_PATH
else:
    for candidate in PROMPT_PATH_FALLBACKS:
        if os.path.exists(candidate):
            _resolved_prompt_path = candidate
            print(f"   (Using fallback path: {candidate})")
            break
if not _resolved_prompt_path:
    print(f"❌ ERROR: Prompt file not found.")
    print(f"   Tried: {PROMPT_PATH}")
    for p in PROMPT_PATH_FALLBACKS:
        print(f"   Tried: {p}")
    print("   Upload your prompt .md file to one of these paths, or set PROMPT_PATH / PROMPT_PATH_FALLBACKS in the CONFIGURATION section.")
    sys.exit(1)

PROMPT_PATH = _resolved_prompt_path
print(f"   Using prompt: {os.path.basename(PROMPT_PATH)}")

# Load prompt
print(f"\n📄 Loading prompt from: {PROMPT_PATH}")
try:
    with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
        MASTER_PROMPT = f.read()
    print(f"✅ Prompt loaded ({len(MASTER_PROMPT)} characters)")
except Exception as e:
    print(f"❌ ERROR: Failed to load prompt file: {e}")
    raise

# Verify prompt content
print("\n🔍 Verifying prompt content...")
if "extracted_names" in MASTER_PROMPT and ("exceptions" in MASTER_PROMPT or "findings" in MASTER_PROMPT or "classification" in MASTER_PROMPT):
    print("✅ Prompt contains expected output instructions")
else:
    print("⚠️  WARNING: Prompt may not contain expected format instructions!")

# Find all JSON files
print(f"\n🔍 Finding JSON files in: {INPUT_DIR}")
if not os.path.exists(INPUT_DIR):
    print(f"❌ ERROR: Input directory does not exist: {INPUT_DIR}")
    print("   On Databricks, set INPUT_DIR to a workspace/DBFS path (e.g. /Workspace/... or /dbfs/...).")
    sys.exit(1)

json_files = find_json_files(INPUT_DIR)
_only = (RUN_ONLY_BASENAME or os.environ.get("PIPELINE_RUN_ONLY", "")).strip()
if _only:
    _before = len(json_files)
    json_files = [f for f in json_files if stem_json_basename(os.path.basename(f)) == _only]
    print(f"🔎 RUN_ONLY_BASENAME={_only!r}: {_before} -> {len(json_files)} file(s)")
    if not json_files:
        print(f"❌ No JSON in INPUT_DIR with stem {_only!r}. Check spelling/case (e.g. Personal_information_9).")
        sys.exit(1)

print(f"✅ Found {len(json_files)} file(s) to process")
if json_files:
    print(f"   Sample files: {[os.path.basename(f) for f in json_files[:5]]}")
else:
    print(f"⚠️  WARNING: No JSON files found!")
    print(f"   Check that files exist and match patterns: *.pdf.json, *.json")

# Create output directory (timestamped so each run uses a new folder; does not mix with previous runs)
_run_suffix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_subdir = os.path.join(OUTPUT_DIR, f'{CATEGORY_NAME}_{VERSION}_{_run_suffix}')
os.makedirs(output_subdir, exist_ok=True)
print(f"\n📁 Output directory: {output_subdir}")

# Setup LangChain model
print("\n" + "="*80)
print("STEP 1: Initializing LangChain model")
print("="*80)
langchain_model = construct_langchain(MODEL, CREDENTIALS_PATH, TEMPERATURE, MAX_TOKENS, PROJECT_LOCATION, PROJECT_ID, streaming=False)
print(f"✅ LangChain model initialized: {MODEL}")

print("\n" + "="*80)
print(f"STEP 2: Setting up system instruction from {os.path.basename(PROMPT_PATH)}")
print("="*80)
system_instruction = create_cached_model(MASTER_PROMPT)

# Document-level prompts (path contains "document_level"): one LLM pass + post-process PDF banner
DOCUMENT_LEVEL_MODE = "document_level" in (PROMPT_PATH or "").lower()
_use_document_level_hybrid = DOCUMENT_LEVEL_ANALYSIS and not DOCUMENT_LEVEL_MODE

# Process files
print("\n" + "="*80)
print("STEP 3: Processing files")
if DOCUMENT_LEVEL_MODE:
    print("   Document-level mode: single LLM call per file + v3.5-compatible JSON shape")
elif _use_document_level_hybrid:
    if DOCUMENT_SPLIT_CLASSIFICATION:
        print("   Document-level analysis: per separated segment (capped text) + aggregated file label")
    else:
        print("   Document-level analysis: single LLM call per file on full text (DOCUMENT_LEVEL_ANALYSIS=True)")
else:
    print("   Segment mode: one LLM call per segment (DOCUMENT_LEVEL_ANALYSIS=False)")
print("="*80)

success_count = 0
failed_files = []  # (basename, reason)

for i, json_file in enumerate(json_files, 1):
    print(f"\n📄 Processing {i}/{len(json_files)}: {os.path.basename(json_file)}")
    
    base_name = os.path.basename(json_file)
    if base_name.endswith('.pdf.json'):
        base_name = base_name.replace('.pdf.json', '')
    elif base_name.endswith('.json'):
        base_name = base_name.replace('.json', '')
    
    output_filename = os.path.join(output_subdir, f"{base_name}_WORKING_PageNumbers_analysis.json")
    
    try:
        processor = DocumentProcessor(
            json_file, 
            langchain_model=langchain_model, 
            system_instruction=system_instruction,
            document_level_mode=DOCUMENT_LEVEL_MODE,
            document_level_analysis=_use_document_level_hybrid,
        )
        result = processor.process_document(output_filepath=output_filename)
        if result is None:
            print(f"❌ Failed (no model output): {os.path.basename(json_file)}")
            failed_files.append((os.path.basename(json_file), "no model output"))
            continue
        print(f"✅ Success: {os.path.basename(output_filename)}")
        success_count += 1
    except Exception as e:
        print(f"❌ Error: {e}")
        failed_files.append((os.path.basename(json_file), repr(e)))
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print(f"✅ COMPLETE! Processed {success_count}/{len(json_files)} files")
print(f"📁 Output directory: {output_subdir}")
if failed_files:
    print(f"\n⚠️  Failed files ({len(failed_files)}):")
    for fn, reason in failed_files:
        print(f"   - {fn}: {reason}")
    _fail_log = os.path.join(output_subdir, "failed_files.txt")
    with open(_fail_log, "w", encoding="utf-8") as _f:
        for fn, reason in failed_files:
            _f.write(f"{fn}\t{reason}\n")
    print(f"   (also saved to {_fail_log})")
print(f"{'='*80}")
print("ℹ️  PDF highlighting removed in this pipeline version.")

