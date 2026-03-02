# DocExtract AI — Agentic Document Extraction Prototype

A proof-of-concept pipeline that uses **LangGraph** to extract structured data from invoice documents (PDF or image) and present the results through a local web UI

---

## Overview

Traditional OCR tools read characters but lose the document's meaning — they can't tell a vendor address from a buyer address, or understand that a table of figures rolls up into a subtotal. This prototype takes a different approach: a chain of specialised AI agents, each responsible for one cognitive step, working together to produce a rich, structured JSON output.

The full pipeline runs locally. The web UI uploads a file, streams live stage updates from the backend, and renders the extracted data as a dashboard.

---

## Architecture

```

                       LangGraph Pipeline                          
                                                                   
   [PDF / Image]                                                                                                                 
                                                        
   1. PREPROCESS     pdfplumber · PyMuPDF · PIL                   
                     Extract text, geometry, page metadata                                                                        
   2. LAYOUT ANALYSIS  LLM                                         
                     Identify structural blocks (header, table,   
                     address block, footer, watermark …)                                                   
   3. ELEMENT CLASSIFY  LLM                                        
                     Assign semantic roles (vendor_info,          
                     invoice_metadata, approval_stamp …)               
   4. CONTENT EXTRACT  LLM                                         
                     Pull typed field values into a strict        
                     schema (vendor, buyer, line items …)                                                             
   5. RELATIONSHIP EXTRACT  LLM                                    
                     Discover implicit links between elements     
                     (issued_by, billed_to, governed_by …)                                                           
   6. RECONSTRUCT     Deterministic                                
                      Merge all stage outputs -> final JSON         

```

The **Flask server** monkey-patches the pipeline's internal logger so every stage update is forwarded to the browser in real time via **Server-Sent Events (SSE)** — no polling required.

---

## Project Structure

```
project/
├── document_extraction_agent.py   # LangGraph pipeline (core engine)
├── test.py                        # CLI demo — mock mode or live LLM
├── server.py                      # Flask backend — REST + SSE
├── templates/
    └── index.html                 # Single-file web UI
├── sample_invoice.pdf             # Bundled test document (https://github.com/ssukhpinder/AzureOpenAI/blob/main/samples/Azure.OpenAI.DocumentIntelligence/sample-document/sample-invoice.pdf)
└── README.md
```

---

## Prerequisites

- Python 3.10+
- An LLM API key — OpenAI (`gpt-4o`)

---

## Running the Web UI

# Create a .env file with your OpenAI API Key
OPENAI_API_KEY=sk-...

```bash

python server.py
```

Open **http://localhost:5000** in your browser.

1. Drop a PDF or image onto the upload zone (or click to browse).
2. Choose a document type hint from the dropdown.
3. Click **Run Extraction**.
4. Watch each pipeline stage light up in real time on the left panel.
5. Results appear as a structured dashboard on the right.

---

## Running Without a UI (CLI / Mock Mode)

```bash
# Mock mode — no API key needed, fires all 6 LangGraph nodes end-to-end
python test.py

# Live mode — OpenAI
python test.py --live
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Serves the web UI |
| `POST` | `/api/extract` | Upload a file, start a pipeline job. Returns `{ job_id }` |
| `GET`  | `/api/stream/<job_id>` | SSE stream of live stage events |
| `GET`  | `/api/result/<job_id>` | Fallback poll — returns the final JSON when ready |

**Upload parameters (multipart/form-data)**

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | Document to extract (PDF, PNG, JPG, TIFF) |
| `doc_type` | String | Hint: `invoice`, `purchase_order`, `receipt`, `contract`, `auto` |

---

## Output JSON Schema

```jsonc
{
  "schema_version": "1.0.0",
  "extraction_metadata": { "source_file", "pages_processed", "errors", … },
  "layout": {
    "structural_blocks": [
      { "block_id": 1, "type": "header", "position": "top", "summary": "…" }
    ]
  },
  "semantics": {
    "classified_elements": [
      { "element_id": 1, "semantic_role": "vendor_info", "confidence": "high", … }
    ]
  },
  "extracted_data": {
    "vendor":      { "name", "address", "phone", "email", "ein", "registration_number" },
    "buyer":       { "name", "contact_name", "address", "client_id" },
    "invoice":     { "number", "date", "due_date", "purchase_order", "payment_terms" },
    "line_items":  [ { "description", "quantity", "unit_price", "tax_rate", "amount" } ],
    "financials":  { "subtotal", "tax_amount", "discount", "total_due", "currency" },
    "payment":     { "bank_name", "account_name", "account_number", "routing", "swift" },
    "legal_notes": { "msa_reference", "late_fee_policy", "disclaimer", "approval_status" }
  },
  "relationships": [
    { "subject": "invoice.line_items", "predicate": "summarised_by",
      "object": "invoice.financials.subtotal", "evidence": "…" }
  ],
  "agent_log": [ "[PREPROCESS] Done – 1 page(s), 1502 chars extracted", … ]
}
```

---

## Extending the Pipeline

| Goal | Where to change |
|------|----------------|
| Support a new document type | Update the system prompts in Nodes 2–4 of `document_extraction_agent.py` |
| Add a validation agent | Insert a new `StateGraph` node between `reconstruct` and `END` |
| Add OCR for scanned images | Install `pytesseract` or `easyocr`; call inside `node_preprocess` for image file types |
| Change the LLM | Swap the `LLM` variable at the top of `document_extraction_agent.py` |
| Stream intermediate state | Use `pipeline.stream(initial_state)` instead of `pipeline.invoke()` and iterate over events |

---

## Known Limitations

- **No validation agent** — cross-checking extracted values against external sources (e.g. company registries, bank whitelists) is not yet implemented.
- **Scanned images** — preprocessing applies basic contrast/sharpening via PIL, but full OCR for handwritten or very low-quality scans requires an additional OCR integration.
- **Large documents** — multi-page PDFs are fully extracted but the LLM context window limits how much text can be analysed in a single call. For documents over ~20 pages, chunking logic should be added.
- **Concurrency** — the Flask server stores jobs in memory; restarting the server clears all running jobs. A production deployment would use a task queue (Celery, RQ) and persistent storage.

---

Prototype / internal use. Not intended for production deployment without additional security review, error handling, and infrastructure hardening.
