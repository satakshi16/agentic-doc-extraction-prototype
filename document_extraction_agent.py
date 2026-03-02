"""
AGENTIC DOCUMENT EXTRACTION            
                                                                              
  Pipeline stages:                                                            
   1. Document Capture & Preprocessing                                        
   2. Layout Analysis                                                         
   3. Element Classification                                                  
   4. Content Extraction                                                      
   5. Relationship Extraction                                                 
   6. Semantic Reconstruction (JSON output)                                   
"""

import os
import re
import json
import time
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import pdfplumber
import fitz                          # PyMuPDF – page-level geometry
from PIL import Image, ImageFilter, ImageEnhance

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from dotenv import load_dotenv

load_dotenv()

LLM = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)

#  Shared State  (passed between every LangGraph node)

class DocState(TypedDict):
    # Inputs
    file_path: str
    doc_type_hint: Optional[str]          # e.g. "invoice", "contract" – optional

    # Stage 1 – Preprocessing
    raw_text: str                          # pdfplumber full text
    page_metadata: List[Dict]             # per-page dims, char-count, table-count
    preprocessed_text: str                # cleaned/normalised text

    # Stage 2 – Layout Analysis
    layout_report: str                    # LLM analysis of structural blocks
    structural_blocks: List[Dict]         # [{block_id, type, bbox_hint, content}]

    # Stage 3 – Element Classification
    classified_elements: List[Dict]       # [{element_id, semantic_role, content}]

    # Stage 4 – Content Extraction
    extracted_fields: Dict[str, Any]      # raw key-value extraction

    # Stage 5 – Relationship Extraction
    relationships: List[Dict]             # [{parent, child, relation_type}]

    # Stage 6 – Semantic Reconstruction
    final_json: Dict[str, Any]            # final structured output

    # Meta
    agent_log: List[str]                  # breadcrumb trail
    errors: List[str]


#  Helper utilities

def _log(state: DocState, agent: str, message: str) -> None:
    state["agent_log"].append(f"[{agent}] {message}")
    print(f"  * [{agent}] {message}")


def _llm_call(system_prompt: str, user_content: str) -> str:
    """Thin wrapper around the LLM with basic retry logic."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    for attempt in range(3):
        try:
            response = LLM.invoke(messages)
            return response.content.strip()
        except Exception as exc:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def _extract_json_block(text: str) -> Any:
    """Pull the first JSON object or array from an LLM response string."""
    # Try to find ```json ... ``` fences first
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if match:
        return json.loads(match.group(1))
    # Fallback: first { ... } block
    match = re.search(r"(\{[\s\S]+\}|\[[\s\S]+\])", text)
    if match:
        return json.loads(match.group(1))
    raise ValueError(f"No JSON found in LLM output:\n{text[:400]}")


#  NODE 1 – Document Capture & Preprocessing

def node_preprocess(state: DocState) -> DocState:
    """
    Ingest the file, correct skew/contrast via PIL, extract raw text and
    per-page geometry metadata using pdfplumber + PyMuPDF.
    """
    _log(state, "PREPROCESS", f"Ingesting: {state['file_path']}")
    path = Path(state["file_path"])
    pages_meta = []

    # PDF path (text-native or scanned) 
    if path.suffix.lower() == ".pdf":
        # Geometry via PyMuPDF
        doc_mu = fitz.open(str(path))
        for page in doc_mu:
            pages_meta.append({
                "page": page.number + 1,
                "width_pt": round(page.rect.width, 1),
                "height_pt": round(page.rect.height, 1),
                "rotation": page.rotation,
            })
        doc_mu.close()

        # Text extraction via pdfplumber
        full_text_parts = []
        with pdfplumber.open(str(path)) as pdf:
            for idx, pg in enumerate(pdf.pages):
                txt = pg.extract_text() or ""
                tables = pg.extract_tables() or []
                # Serialise tables into a readable block
                table_text = ""
                for tbl_idx, tbl in enumerate(tables):
                    table_text += f"\n[TABLE {idx+1}.{tbl_idx+1}]\n"
                    for row in tbl:
                        row_clean = [str(c).strip() if c else "" for c in row]
                        table_text += " | ".join(row_clean) + "\n"
                if idx < len(pages_meta):
                    pages_meta[idx]["char_count"] = len(txt)
                    pages_meta[idx]["table_count"] = len(tables)
                full_text_parts.append(f"=== PAGE {idx+1} ===\n{txt}{table_text}")

        state["raw_text"] = "\n".join(full_text_parts)

    # Image path – apply basic preprocessing with PIL
    elif path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        img = Image.open(str(path)).convert("L")  # grayscale
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = img.filter(ImageFilter.SHARPEN)
        preprocessed_path = path.with_stem(path.stem + "_preprocessed")
        img.save(preprocessed_path)
        # For images we'd normally OCR here; we'll put a placeholder
        state["raw_text"] = f"[IMAGE FILE – preprocessed copy at {preprocessed_path}]"
        pages_meta = [{"page": 1, "width_pt": img.width, "height_pt": img.height}]
        _log(state, "PREPROCESS", "Image preprocessing applied (contrast + sharpen)")

    else:
        with open(path, "r", errors="ignore") as f:
            state["raw_text"] = f.read()
        pages_meta = [{"page": 1, "char_count": len(state["raw_text"])}]

    state["page_metadata"] = pages_meta

    # Light text normalisation
    cleaned = state["raw_text"]
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)           # collapse blank lines
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)            # collapse spaces
    cleaned = cleaned.strip()
    state["preprocessed_text"] = cleaned

    _log(state, "PREPROCESS",
         f"Done – {len(pages_meta)} page(s), {len(cleaned)} chars extracted")
    return state


#  NODE 2 – Layout Analysis

LAYOUT_SYSTEM = """You are a document layout analysis agent.
Given raw extracted text from a document, identify its high-level structural 
blocks. For each block produce a JSON array where every item has:
  block_id   : integer starting at 1
  type       : one of [header, subheader, metadata_block, table, line_item_row,
                       totals_block, address_block, notes_block, 
                       footer_disclaimer, watermark_text, other]
  position   : "top" | "upper" | "middle" | "lower" | "bottom"
  summary    : one-sentence description of what this block contains

Respond ONLY with valid JSON (no markdown prose outside the code block)."""

def node_layout_analysis(state: DocState) -> DocState:
    _log(state, "LAYOUT", "Analysing document structure …")

    prompt_user = (
        f"Document type hint: {state.get('doc_type_hint') or 'unknown'}\n\n"
        f"Extracted text:\n{state['preprocessed_text'][:6000]}"
    )

    raw_response = _llm_call(LAYOUT_SYSTEM, prompt_user)
    state["layout_report"] = raw_response

    try:
        blocks = _extract_json_block(raw_response)
        if isinstance(blocks, dict):
            blocks = blocks.get("blocks", [blocks])
        state["structural_blocks"] = blocks
        _log(state, "LAYOUT", f"Identified {len(blocks)} structural blocks")
    except Exception as exc:
        state["structural_blocks"] = []
        state["errors"].append(f"Layout JSON parse error: {exc}")
        _log(state, "LAYOUT", f"⚠ JSON parse failed: {exc}")

    return state


#  NODE 3 – Element Classification

CLASSIFY_SYSTEM = """You are a semantic element classification agent.
Given structural blocks from a document, assign a fine-grained semantic role to 
each. Produce a JSON array where every item has:
  element_id    : integer
  block_id      : integer (links back to the layout block)
  semantic_role : one of [document_title, vendor_info, buyer_info, 
                          invoice_metadata, line_item, financial_summary,
                          payment_instructions, legal_disclaimer, 
                          approval_stamp, company_registration, other]
  confidence    : "high" | "medium" | "low"
  content_snippet: first 120 chars of the block's content

Respond ONLY with valid JSON."""

def node_classify_elements(state: DocState) -> DocState:
    _log(state, "CLASSIFY", "Assigning semantic roles to blocks …")

    blocks_json = json.dumps(state["structural_blocks"], indent=2)
    prompt_user = (
        f"Document blocks:\n{blocks_json}\n\n"
        f"Full text reference:\n{state['preprocessed_text'][:4000]}"
    )

    raw_response = _llm_call(CLASSIFY_SYSTEM, prompt_user)

    try:
        elements = _extract_json_block(raw_response)
        if isinstance(elements, dict):
            elements = elements.get("elements", [elements])
        state["classified_elements"] = elements
        _log(state, "CLASSIFY",
             f"Classified {len(elements)} elements")
    except Exception as exc:
        state["classified_elements"] = []
        state["errors"].append(f"Classify JSON parse error: {exc}")
        _log(state, "CLASSIFY", f"⚠ JSON parse failed: {exc}")

    return state


#  NODE 4 – Content Extraction

EXTRACT_SYSTEM = """You are a precision content-extraction agent.
Given classified elements and the full document text, extract ALL meaningful 
field values. Produce a single JSON object with these top-level keys 
(use null if a value is genuinely absent):

  vendor        : { name, address, phone, email, ein, registration_number }
  buyer         : { name, contact_name, address, client_id }
  invoice       : { number, date, due_date, purchase_order, payment_terms }
  line_items    : [ { description, quantity, unit_price, tax_rate, amount } ]
  financials    : { subtotal, tax_amount, discount, total_due, currency }
  payment       : { bank_name, account_name, account_number, routing, swift }
  legal_notes   : { msa_reference, late_fee_policy, disclaimer, approval_status }
  document_meta : { pages, file_name, doc_type }

Respond ONLY with valid JSON – no extra prose."""

def node_extract_content(state: DocState) -> DocState:
    _log(state, "EXTRACT", "Extracting typed field values …")

    elements_json = json.dumps(state["classified_elements"], indent=2)
    prompt_user = (
        f"Classified elements:\n{elements_json}\n\n"
        f"Full document text:\n{state['preprocessed_text']}"
    )

    raw_response = _llm_call(EXTRACT_SYSTEM, prompt_user)

    try:
        fields = _extract_json_block(raw_response)
        state["extracted_fields"] = fields
        total = sum(
            1 for v in fields.values()
            if v and v != {} and v != []
        )
        _log(state, "EXTRACT", f"Populated {total} top-level field groups")
    except Exception as exc:
        state["extracted_fields"] = {}
        state["errors"].append(f"Extract JSON parse error: {exc}")
        _log(state, "EXTRACT", f"⚠ JSON parse failed: {exc}")

    return state


#  NODE 5 – Relationship Extraction

RELATION_SYSTEM = """You are a document-relationship extraction agent.
Analyse the document structure and extracted content to surface implicit 
relationships between elements. Produce a JSON array where each item has:
  relation_id  : integer
  subject      : string – what (e.g. "invoice.line_items")
  predicate    : string – relationship type (e.g. "belongs_to", "summarised_by",
                  "authorised_by", "governed_by", "billed_to", "issued_by")
  object       : string – the related item (e.g. "invoice.financials.subtotal")
  evidence     : one-sentence justification from the document text

Focus on non-obvious relationships that would be useful for downstream 
processing. Respond ONLY with valid JSON."""

def node_extract_relationships(state: DocState) -> DocState:
    _log(state, "RELATIONS", "Mining element relationships …")

    fields_json = json.dumps(state["extracted_fields"], indent=2)
    elements_json = json.dumps(state["classified_elements"][:10], indent=2)  # top 10

    prompt_user = (
        f"Extracted fields:\n{fields_json}\n\n"
        f"Classified elements (sample):\n{elements_json}"
    )

    raw_response = _llm_call(RELATION_SYSTEM, prompt_user)

    try:
        rels = _extract_json_block(raw_response)
        if isinstance(rels, dict):
            rels = rels.get("relationships", [rels])
        state["relationships"] = rels
        _log(state, "RELATIONS", f"Found {len(rels)} relationships")
    except Exception as exc:
        state["relationships"] = []
        state["errors"].append(f"Relations JSON parse error: {exc}")
        _log(state, "RELATIONS", f"⚠ JSON parse failed: {exc}")

    return state


#  NODE 6 – Semantic Reconstruction

def node_reconstruct(state: DocState) -> DocState:
    """
    Assemble all agent outputs into the canonical final JSON.
    This node does NOT call the LLM – it merges deterministically.
    """
    _log(state, "RECONSTRUCT", "Building final structured JSON …")

    final = {
        "schema_version": "1.0.0",
        "extraction_metadata": {
            "source_file": state["file_path"],
            "document_type": state.get("doc_type_hint") or "auto-detected",
            "pages_processed": len(state["page_metadata"]),
            "page_details": state["page_metadata"],
            "pipeline_stages": [
                "preprocessing",
                "layout_analysis",
                "element_classification",
                "content_extraction",
                "relationship_extraction",
                "semantic_reconstruction",
            ],
            "errors": state["errors"],
        },
        "layout": {
            "structural_blocks_count": len(state["structural_blocks"]),
            "structural_blocks": state["structural_blocks"],
        },
        "semantics": {
            "classified_elements_count": len(state["classified_elements"]),
            "classified_elements": state["classified_elements"],
        },
        "extracted_data": state["extracted_fields"],
        "relationships": state["relationships"],
        "agent_log": state["agent_log"],
    }

    state["final_json"] = final
    _log(state, "RECONSTRUCT", "Final JSON assembled successfully ✓")
    return state


#  Build the LangGraph  StateGraph

def build_graph() -> Any:
    graph = StateGraph(DocState)

    # Register nodes
    graph.add_node("preprocess", node_preprocess)
    graph.add_node("layout_analysis", node_layout_analysis)
    graph.add_node("classify_elements", node_classify_elements)
    graph.add_node("extract_content", node_extract_content)
    graph.add_node("extract_relationships", node_extract_relationships)
    graph.add_node("reconstruct", node_reconstruct)

    # Linear pipeline edges
    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "layout_analysis")
    graph.add_edge("layout_analysis", "classify_elements")
    graph.add_edge("classify_elements", "extract_content")
    graph.add_edge("extract_content", "extract_relationships")
    graph.add_edge("extract_relationships", "reconstruct")
    graph.add_edge("reconstruct", END)

    return graph.compile()


#  Public entry-point

def extract_document(
    file_path: str,
    doc_type_hint: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full agentic extraction pipeline on *file_path* and return the
    final structured JSON dict.

    Parameters
    ----------
    file_path   : path to a PDF, image, or text file
    doc_type_hint : optional hint ("invoice", "contract", "medical_form" …)
    output_path : if given, write JSON to this file

    Returns
    -------
    dict  – the complete extraction result
    """
    print("\n" + "#" * 70)
    print("  AGENTIC DOCUMENT EXTRACTION  –  LangGraph Pipeline")
    print("#" * 70)
    print(f"  File : {file_path}")
    print(f"  Type : {doc_type_hint or 'auto-detect'}")
    print("─" * 70)

    initial_state: DocState = {
        "file_path": file_path,
        "doc_type_hint": doc_type_hint,
        "raw_text": "",
        "page_metadata": [],
        "preprocessed_text": "",
        "layout_report": "",
        "structural_blocks": [],
        "classified_elements": [],
        "extracted_fields": {},
        "relationships": [],
        "final_json": {},
        "agent_log": [],
        "errors": [],
    }

    pipeline = build_graph()
    final_state = pipeline.invoke(initial_state)

    result = final_state["final_json"]

    # Write output 
    if output_path is None:
        stem = Path(file_path).stem
        output_path = str(Path(file_path).parent / f"{stem}_extracted.json")

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    print("─" * 70)
    errors = final_state.get("errors", [])
    if errors:
        print(f"  ⚠ Pipeline completed with {len(errors)} error(s):")
        for e in errors:
            print(f"    • {e}")
    else:
        print("  ✅ Pipeline completed with no errors")
    print(f"  📄 Output written → {output_path}")
    print("═" * 70 + "\n")

    return result


#  CLI demo

if __name__ == "__main__":
    import sys

    pdf = sys.argv[1] if len(sys.argv) > 1 else "sample_invoice.pdf"
    hint = sys.argv[2] if len(sys.argv) > 2 else "invoice"

    result = extract_document(pdf, doc_type_hint=hint)

    # Pretty-print just the extracted_data section for a quick view
    print("\n------------------------ EXTRACTED DATA SNAPSHOT ------------------------")
    print(json.dumps(result.get("extracted_data", {}), indent=2))
    print("\n------------------------ RELATIONSHIPS ------------------------")
    for r in result.get("relationships", [])[:6]:
        print(f"  {r.get('subject')}  –[{r.get('predicate')}]→  {r.get('object')}")