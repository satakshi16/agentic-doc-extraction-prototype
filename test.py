"""
demo_runner.py
--------------
Two modes:

  1. MOCK MODE  (default, no API key needed)
     Runs the full LangGraph pipeline end-to-end using a lightweight 
     local mock LLM so you can see every stage fire without spending tokens.

  2. LIVE MODE
     Swaps in the real LLM and runs against the sample invoice.

Usage:
    python test.py               # mock mode
    python test.py --live
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv

load_dotenv()

#  Mock LLM responses – one per pipeline stage

MOCK_LAYOUT = json.dumps([
    {"block_id": 1, "type": "header",            "position": "top",    "summary": "Company name INVOICE banner with vendor branding"},
    {"block_id": 2, "type": "metadata_block",    "position": "upper",  "summary": "Invoice number, date, due date, PO number"},
    {"block_id": 3, "type": "address_block",     "position": "upper",  "summary": "Vendor address and contact details"},
    {"block_id": 4, "type": "address_block",     "position": "upper",  "summary": "Buyer Bill-To address block with client ID"},
    {"block_id": 5, "type": "table",             "position": "middle", "summary": "Line-item table with description, qty, price, tax, amount"},
    {"block_id": 6, "type": "totals_block",      "position": "lower",  "summary": "Subtotal, tax, discount and TOTAL DUE amounts"},
    {"block_id": 7, "type": "notes_block",       "position": "lower",  "summary": "Payment bank details and terms"},
    {"block_id": 8, "type": "notes_block",       "position": "lower",  "summary": "Service notes referencing MSA agreement"},
    {"block_id": 9, "type": "watermark_text",    "position": "middle", "summary": "APPROVED stamp overlaid on document body"},
    {"block_id": 10,"type": "footer_disclaimer", "position": "bottom", "summary": "Legal disclaimer, company registration, EIN, page number"},
], indent=2)

MOCK_CLASSIFY = json.dumps([
    {"element_id":1,  "block_id":1,  "semantic_role":"document_title",          "confidence":"high",   "content_snippet":"INVOICE"},
    {"element_id":2,  "block_id":1,  "semantic_role":"vendor_info",              "confidence":"high",   "content_snippet":"Acme Solutions Inc. – 123 Business Ave"},
    {"element_id":3,  "block_id":2,  "semantic_role":"invoice_metadata",         "confidence":"high",   "content_snippet":"INV-2024-00847 | March 15, 2024 | Due April 14"},
    {"element_id":4,  "block_id":3,  "semantic_role":"vendor_info",              "confidence":"high",   "content_snippet":"Tel: +1 (415) 555-0199 | billing@acmesolutions.com"},
    {"element_id":5,  "block_id":4,  "semantic_role":"buyer_info",               "confidence":"high",   "content_snippet":"TechCorp Global Ltd. – Attn: John Smith, CLT-00293"},
    {"element_id":6,  "block_id":5,  "semantic_role":"line_item",                "confidence":"high",   "content_snippet":"Cloud Infrastructure Setup & Config – 1 × $3,500.00"},
    {"element_id":7,  "block_id":6,  "semantic_role":"financial_summary",        "confidence":"high",   "content_snippet":"TOTAL DUE: $20,006.81"},
    {"element_id":8,  "block_id":7,  "semantic_role":"payment_instructions",     "confidence":"high",   "content_snippet":"First National Bank – Account No: 1234567890"},
    {"element_id":9,  "block_id":8,  "semantic_role":"legal_disclaimer",         "confidence":"medium", "content_snippet":"Services rendered per MSA-2023-112"},
    {"element_id":10, "block_id":9,  "semantic_role":"approval_stamp",           "confidence":"high",   "content_snippet":"APPROVED stamp visible on document"},
    {"element_id":11, "block_id":10, "semantic_role":"company_registration",     "confidence":"high",   "content_snippet":"CA Sec of State #4829301 | EIN: 94-3827650"},
], indent=2)

MOCK_EXTRACT = json.dumps({
    "vendor": {
        "name": "Acme Solutions Inc.",
        "address": "123 Business Ave, San Francisco, CA 94102",
        "phone": "+1 (415) 555-0199",
        "email": "billing@acmesolutions.com",
        "ein": "94-3827650",
        "registration_number": "CA Sec of State #4829301"
    },
    "buyer": {
        "name": "TechCorp Global Ltd.",
        "contact_name": "John Smith",
        "address": "456 Enterprise Blvd, Suite 200, New York, NY 10001",
        "client_id": "CLT-00293"
    },
    "invoice": {
        "number": "INV-2024-00847",
        "date": "2024-03-15",
        "due_date": "2024-04-14",
        "purchase_order": "PO-7823",
        "payment_terms": "Net 30"
    },
    "line_items": [
        {"description": "Cloud Infrastructure Setup & Config",          "quantity": 1,  "unit_price": 3500.00,  "tax_rate": "8.5%", "amount": 3797.50},
        {"description": "Software License - Enterprise Suite (Annual)", "quantity": 3,  "unit_price": 1200.00,  "tax_rate": "8.5%", "amount": 3898.80},
        {"description": "Technical Consulting (40 hrs @ $150/hr)",      "quantity": 40, "unit_price": 150.00,   "tax_rate": "0.0%", "amount": 6000.00},
        {"description": "API Integration Services",                     "quantity": 1,  "unit_price": 2250.00,  "tax_rate": "8.5%", "amount": 2441.25},
        {"description": "Training Workshop – 2 day onsite",             "quantity": 1,  "unit_price": 4000.00,  "tax_rate": "0.0%", "amount": 4000.00},
        {"description": "Support & Maintenance Package (Q1)",           "quantity": 1,  "unit_price": 850.00,   "tax_rate": "8.5%", "amount": 922.25},
    ],
    "financials": {
        "subtotal":     20200.00,
        "tax_amount":   859.80,
        "discount":     -1052.99,
        "total_due":    20006.81,
        "currency":     "USD"
    },
    "payment": {
        "bank_name":       "First National Bank",
        "account_name":    "Acme Solutions Inc.",
        "account_number":  "1234567890",
        "routing":         "021000021",
        "swift":           "FNBKUS33"
    },
    "legal_notes": {
        "msa_reference":   "MSA-2023-112 dated January 10, 2023",
        "late_fee_policy": "1.5% per month after due date",
        "disclaimer":      "Invoice generated electronically; valid without physical signature.",
        "approval_status": "APPROVED"
    },
    "document_meta": {
        "pages":     1,
        "file_name": "sample_invoice.pdf",
        "doc_type":  "invoice"
    }
}, indent=2)

MOCK_RELATIONS = json.dumps([
    {"relation_id":1,  "subject":"invoice.line_items",              "predicate":"summarised_by",  "object":"invoice.financials.subtotal",       "evidence":"Six line items sum to the subtotal of $20,200.00 shown in the totals block."},
    {"relation_id":2,  "subject":"invoice",                         "predicate":"issued_by",       "object":"vendor",                            "evidence":"INVOICE header is associated with Acme Solutions Inc. address and contact details."},
    {"relation_id":3,  "subject":"invoice",                         "predicate":"billed_to",       "object":"buyer",                             "evidence":"'Bill To' section explicitly names TechCorp Global Ltd. as recipient."},
    {"relation_id":4,  "subject":"invoice",                         "predicate":"governed_by",     "object":"legal_notes.msa_reference",         "evidence":"Notes state: services rendered per Master Service Agreement #MSA-2023-112."},
    {"relation_id":5,  "subject":"invoice.financials.total_due",    "predicate":"includes",        "object":"invoice.financials.tax_amount",      "evidence":"Totals block shows tax line contributing to the final TOTAL DUE."},
    {"relation_id":6,  "subject":"invoice.financials.total_due",    "predicate":"reduced_by",      "object":"invoice.financials.discount",        "evidence":"A 5% discount of $1,052.99 is deducted before arriving at TOTAL DUE."},
    {"relation_id":7,  "subject":"invoice",                         "predicate":"authorised_by",   "object":"legal_notes.approval_status",       "evidence":"APPROVED watermark stamp is overlaid on the document body."},
    {"relation_id":8,  "subject":"invoice",                         "predicate":"payable_via",     "object":"payment",                           "evidence":"Payment Instructions block provides bank, account, routing and SWIFT details."},
    {"relation_id":9,  "subject":"invoice.line_items[2]",           "predicate":"detail_of",       "object":"invoice.line_items[2].unit_price",  "evidence":"Technical Consulting qty×rate: 40 hrs × $150 = $6,000 as shown in table."},
    {"relation_id":10, "subject":"legal_notes.late_fee_policy",     "predicate":"applies_after",   "object":"invoice.due_date",                  "evidence":"Late fee of 1.5%/month activates after the Net 30 due date of April 14, 2024."},
], indent=2)


#  Mock LLM class

class MockLLM:
    """
    Round-robin mock: successive calls return LAYOUT → CLASSIFY → EXTRACT →
    RELATIONS responses.  Wraps each in a fake AIMessage so the pipeline
    code sees the same interface as a real ChatOpenAI.
    """
    _responses = [MOCK_LAYOUT, MOCK_CLASSIFY, MOCK_EXTRACT, MOCK_RELATIONS]
    _call_count = 0

    def invoke(self, messages):
        resp = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        obj = MagicMock()
        obj.content = f"```json\n{resp}\n```"
        return obj


#  Runner

def run_mock(input_filename: str):
    """Run the full LangGraph pipeline with mock LLM responses."""
    print("\n" + "#" * 70)
    print("  MOCK MODE  –  no API key required")
    print("  The pipeline fires every LangGraph node; LLM calls return")
    print("  pre-built realistic responses for the sample invoice.")
    print("#" * 70)

    # Patch the module-level LLM before importing
    import document_extraction_agent as agent
    agent.LLM = MockLLM()

    pdf = str(Path(__file__).parent / input_filename)
    out = str(Path(__file__).parent / "sample_invoice_extracted_MOCK.json")
    result = agent.extract_document(pdf, doc_type_hint="invoice", output_path=out)
    return result, out


def run_live(input_filename: str):
    """Run the full pipeline with a real LLM."""
    print("\n" + "#" * 70)
    print(f"  LIVE MODE  ")
    print("#" * 70)

    import document_extraction_agent as agent

    from langchain_openai import ChatOpenAI
    agent.LLM = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
    )

    pdf = str(Path(__file__).parent / input_filename)
    out = str(Path(__file__).parent / "sample_invoice_extracted_LIVE.json")
    result = agent.extract_document(pdf, doc_type_hint="invoice", output_path=out)
    return result, out

#  Safe format
def safe_format(value, width=52, default="–"):
    """
    Safely format any value for printing.
    - Handles None
    - Converts non-strings to string
    - Truncates safely
    """
    if value is None:
        value = default
    else:
        value = str(value)

    return f"{value[:width]:<{width}}"
#  Pretty printer

def pretty_summary(result: dict):
    ed = result.get("extracted_data", {})

    print("\n")
    print("                  EXTRACTION SUMMARY                             ")
    print("")

    vendor = ed.get("vendor", {})
    print(f"   Vendor          : {safe_format(vendor.get('name'))}")
    print(f"   Vendor EIN      : {safe_format(vendor.get('ein'))}")

    buyer = ed.get("buyer", {})
    print(f"   Buyer           : {safe_format(buyer.get('name'))}") 
    print(f"   Client ID       : {safe_format(buyer.get('client_id'))}")

    inv = ed.get("invoice", {})
    print(f"   Invoice #       : {safe_format(inv.get('number'))}") 
    print(f"   Invoice Date    : {safe_format(inv.get('date'))}") 
    print(f"   Due Date        : {safe_format(inv.get('due_date'))}") 
    print(f"   PO #            : {safe_format(inv.get('purchase_order'))}") 

    fin = ed.get("financials", {})
    print(f"   Subtotal        : {safe_format(str(fin.get('subtotal')))}")  
    print(f"   Tax             : {safe_format(str(fin.get('tax_amount')))}")
    print(f"   Discount        : {safe_format(str(fin.get('discount')))}")
    print(f"   TOTAL DUE       : {safe_format(str(fin.get('total_due')))}")
    print(f"   Currency        : {safe_format(fin.get('currency'))}") 

    legal = ed.get("legal_notes", {})
    print(f"   Approval Status : {safe_format(legal.get('approval_status'))}")
    print(f"   MSA Ref         : {safe_format(legal.get('msa_reference'))}")

    line_items = ed.get("line_items", [])
    print(f"   Line Items      : {str(len(line_items)) + ' items extracted':<52} ")

    rels = result.get("relationships", [])
    print(f"   Relationships   : {str(len(rels)) + ' discovered':<52} ")

    blocks = result.get("layout", {}).get("structural_blocks_count", 0)
    print(f"   Layout Blocks   : {str(blocks):<52} ")

    errors = result.get("extraction_metadata", {}).get("errors", [])
    status = "✅ CLEAN" if not errors else f"⚠ {len(errors)} error(s)"
    print(f"   Pipeline Status : {status:<52} ")
    print("-----------------------------------------------------------------")

    if line_items:
        print("\n  Line Items Extracted:")
        print(f"  {'Description':<45} {'Qty':>5}  {'Unit $':>10}  {'Amount':>10}")
        print("  " + "─" * 75)
        for li in line_items:
            desc = str(li.get("description",""))[:44]
            print(f"  {desc:<45} {str(li.get('quantity',''))[:5]:>5}  "
                  f"{'$'+str(li.get('unit_price','')):>10}  "
                  f"{'$'+str(li.get('amount','')):>10}")

    if rels:
        print("\n  Key Relationships:")
        for r in rels[:6]:
            print(f"  • {r.get('subject')}  –[{r.get('predicate')}]→  {r.get('object')}")


#  Entry point

if __name__ == "__main__":
    live = "--live" in sys.argv

    input_filename = "wholefoods_20240528_002.pdf"
    if live:
        result, out_path = run_live(input_filename)
    else:
        result, out_path = run_mock(input_filename)

    pretty_summary(result)
    print(f"\n  Full JSON saved → {out_path}\n")