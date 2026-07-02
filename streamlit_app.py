"""
Bank Statement Analyzer - Streamlit UI
=====================================

Beautiful web interface for your hybrid multi-agent system
"""

import json
import re
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, date
import os
import copy
import tempfile
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from streamlit_problem_solver_v2 import render_problem_solver_page as render_problem_solver_page_v2

try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_AVAILABLE = True
except ImportError:
    PLOTLY_EVENTS_AVAILABLE = False

# Import your system
from main_coordinator import BankStatementAnalyzer
from utils.financial_solver import solve_problem
from utils.llm_interface import resolve_openai_api_key

KNOWLEDGE_BASE_PATH = Path(__file__).resolve().parent / "data" / "advisory_knowledge_base_v1.json"

CATEGORY_CODES = [
    'food_dining',
    'groceries',
    'transportation',
    'shopping',
    'bills_utilities',
    'entertainment',
    'healthcare',
    'income',
    'fees',
    'other',
    'uncategorized',
]

def _category_code_to_label(code: str) -> str:
    return code.replace('_', ' ').title()

def _category_label_to_code(label: str) -> str:
    normalized = str(label).strip().lower().replace(' ', '_')
    if normalized in CATEGORY_CODES:
        return normalized
    return 'other'

def _transactions_to_editor_df(transactions: list) -> pd.DataFrame:
    """Create editable DataFrame for manual category review."""
    rows = []
    for idx, txn in enumerate(transactions):
        rows.append({
            '_txn_index': int(txn.get('_global_index', idx)),
            'Date': txn['date'],
            'Month': txn.get('month', ''),
            'Person': txn.get('person', ''),
            'Document': txn.get('source_file_name', ''),
            'Doc Type': txn.get('document_type', ''),
            'Description': txn['description'],
            'Category': _category_code_to_label(txn['category']),
            'Amount': txn['amount'],
            'Type': 'OUT' if txn['is_debit'] else 'IN',
            'Counts as spending': bool(txn.get('effective_is_spending', txn.get('is_debit', False))),
            'Internal transfer?': bool(txn.get('possible_internal_transfer', False)),
            'Confidence': f"{txn['confidence']:.0%}",
            'Source': txn['source'].title()
        })
    return pd.DataFrame(rows)


def _parse_transaction_date(date_text: str) -> Optional[date]:
    date_text = str(date_text or '').strip()
    if not date_text:
        return None

    date_formats = [
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%d/%m/%Y',
        '%m/%d/%Y',
        '%d-%m-%Y',
        '%m-%d-%Y',
    ]

    for fmt in date_formats:
        try:
            return datetime.strptime(date_text, fmt).date()
        except Exception:
            continue

    try:
        return datetime.fromisoformat(date_text).date()
    except Exception:
        return None


def _filter_transactions(
    transactions: list,
    person_filter: str,
    month_filter: str,
    document_filter: str,
    date_filter_mode: str = "Todos",
    date_range: Optional[tuple[Optional[date], Optional[date]]] = None,
) -> list:
    filtered = []
    for idx, txn in enumerate(transactions):
        if person_filter != "Todos" and txn.get('person') != person_filter:
            continue
        if month_filter != "Todos" and txn.get('month') != month_filter:
            continue
        if document_filter != "Todos" and txn.get('source_file_name') != document_filter:
            continue

        txn_date = _parse_transaction_date(txn.get('date'))
        if date_filter_mode == "Rango de fechas":
            start_date, end_date = date_range or (None, None)
            if txn_date is None or start_date is None or end_date is None:
                continue
            if not (start_date <= txn_date <= end_date):
                continue

        enriched = dict(txn)
        enriched['_global_index'] = idx
        filtered.append(enriched)
    return filtered


def _summarize_transactions(transactions: list) -> dict:
    total_income = 0.0
    total_spent = 0.0
    credit_count = 0
    debit_count = 0

    for txn in transactions:
        amount = float(txn.get('amount') or 0.0)
        if txn.get('is_debit'):
            total_spent += amount
            debit_count += 1
        else:
            total_income += amount
            credit_count += 1

    return {
        'total_spent': total_spent,
        'total_income': total_income,
        'net_change': total_income - total_spent,
        'debit_count': debit_count,
        'credit_count': credit_count,
    }


def _build_statement_inputs_from_uploads(uploaded_files: list) -> list:
    """Capture deterministic per-document metadata from the user before analysis."""
    statement_inputs = []

    st.subheader("🗂️ Metadata por documento")
    st.caption("Estos campos ayudan a consolidar múltiples cartolas con trazabilidad. Si no estás seguro, deja valores por defecto.")

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        with st.expander(f"Documento {idx}: {uploaded_file.name}", expanded=(idx <= 2)):
            col1, col2 = st.columns(2)
            with col1:
                document_type = st.selectbox(
                    "Tipo de documento",
                    ["bank_account", "credit_card", "other"],
                    index=0,
                    key=f"meta_doc_type_{idx}",
                    format_func=lambda x: {
                        "bank_account": "Cuenta bancaria",
                        "credit_card": "Tarjeta de crédito",
                        "other": "Otro",
                    }.get(x, x),
                )
                person = st.text_input(
                    "Persona asociada",
                    value=f"persona_{idx}",
                    key=f"meta_person_{idx}",
                    help="Ejemplo: marido, mujer, conjunta, hijo, etc.",
                ).strip() or f"persona_{idx}"
            with col2:
                account_label = st.text_input(
                    "Etiqueta de cuenta",
                    value=f"cuenta_{idx}",
                    key=f"meta_account_{idx}",
                    help="Ejemplo: cuenta corriente BCI, tarjeta visa banco X",
                ).strip() or f"cuenta_{idx}"
                institution = st.text_input(
                    "Institución (opcional)",
                    value="",
                    key=f"meta_institution_{idx}",
                ).strip()

            statement_inputs.append({
                'uploaded_file': uploaded_file,
                'metadata': {
                    'file_name': uploaded_file.name,
                    'document_type': document_type,
                    'person': person,
                    'account_label': account_label,
                    'institution': institution,
                }
            })

    return statement_inputs

def _get_category_items_for_modal(transactions: list, category_label: str) -> list:
    """Return debit transactions for the selected category, preserving table order."""
    category_code = _category_label_to_code(category_label)
    category_items = []

    for txn in transactions:
        if txn['is_debit'] and txn['category'] == category_code:
            category_items.append({
                'Description': txn['description'],
                'Amount': txn['amount']
            })

    return category_items


def _get_openai_api_key() -> Optional[str]:
    """Resolve the API key from Streamlit secrets first, then local environment fallback.

    Streamlit Cloud should store the key in Secrets. Local development can still use .env.
    """
    return resolve_openai_api_key()


@st.cache_data(show_spinner=False)
def _load_knowledge_base() -> dict:
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_json_response(response_text: str) -> dict:
    cleaned = response_text.strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    raise ValueError("No valid JSON found in response")


def _build_meta_analysis_payload(result: dict) -> dict:
    analysis = result['analysis']
    return {
        'financial_summary': analysis['financial_summary'],
        'category_breakdown': analysis['category_breakdown'],
        'spending_patterns': analysis['spending_patterns'],
        'basic_insights': analysis['basic_insights'],
        'transactions': [
            {
                'date': txn['date'],
                'description': txn['description'],
                'category': txn['category'],
                'amount': txn['amount'],
                'is_debit': txn['is_debit'],
                'confidence': txn['confidence'],
                'source': txn['source']
            }
            for txn in result['transactions']
        ]
    }


def _redact_diagnostic_line(line: str) -> str:
    redacted = str(line or "").strip()
    redacted = re.sub(r'\b\d{4,}\b', '[redacted]', redacted)
    redacted = re.sub(r'(?i)(\$\s*)?\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d{2})', '[amount redacted]', redacted)
    redacted = re.sub(r'\s+', ' ', redacted)
    if len(redacted) > 120:
        redacted = redacted[:117] + '...'
    return redacted


def _run_meta_analysis(result: dict) -> dict:
    knowledge_base = _load_knowledge_base()
    payload = _build_meta_analysis_payload(result)
    compact_knowledge_base = json.dumps(knowledge_base, ensure_ascii=False, separators=(',', ':'))

    system_prompt = f"""You are the second-stage meta advisory engine for a personal finance system.
Use the knowledge base below as the primary policy and evaluation source.

Rules:
- Do not invent data.
- Do not infer income, debt capacity, or intent without explicit evidence.
- Prioritize prudence and risk reduction when uncertainty exists.
- Return ONLY valid JSON matching the output contract in the knowledge base.
- Every recommendation must include evidence.

Knowledge base:
{compact_knowledge_base}
"""

    user_prompt = f"""Run a second-stage meta analysis over the fully categorized statement after human review.
Use the reviewed transactions and the base analysis context.

Base analysis context:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""

    llm = st.session_state.analyzer.llm
    calls_before = llm.call_count
    cost_before = llm.total_cost

    try:
        response = llm.make_call(user_prompt, system_prompt, expect_json=True)
    except RuntimeError as e:
        print(f"First JSON call failed: {e}. Retrying without JSON mode limit...")
        response = None
        
    if not response:
        retry_system_prompt = system_prompt + "\nIf you cannot use JSON mode, still return only valid JSON and no markdown."
        response = llm.make_call(user_prompt, retry_system_prompt, expect_json=False)
        
    if not response:
        raise ValueError("Meta analysis LLM call failed. Could not obtain response from OpenAI.")

    meta_result = _extract_json_response(response)

    # Validate meta_result against knowledge base output contract (basic checks)
    kb_contract = knowledge_base.get('output_contract', {})
    try:
        _validate_meta_output(meta_result, kb_contract)
    except Exception as e:
        raise ValueError(f"Meta analysis result validation failed: {e}")

    calls_used = llm.call_count - calls_before
    cost_used = llm.total_cost - cost_before

    return {
        'result': meta_result,
        'metrics': {
            'llm_calls': max(calls_used, 0),
            'estimated_cost': max(cost_used, 0.0),
            'base_cost': result['system_metrics']['estimated_cost'],
            'total_cost': result['system_metrics']['estimated_cost'] + max(cost_used, 0.0)
        },
        'timestamp': datetime.now().isoformat(),
        'source_signature': result['analysis']['report_generated_at']
    }


def _is_number_like(v):
    try:
        if isinstance(v, (int, float)):
            return True
        float(v)
        return True
    except Exception:
        return False


def _validate_solved_result(solved: dict):
    """Basic validation that the deterministic solver returned numeric results.

    Raises ValueError if critical numeric fields are missing or non-numeric.
    """
    result = solved.get('result') or {}
    if not result:
        raise ValueError('Solver returned empty result')

    # Check common expected numeric fields for different problem types
    # We only enforce that all leaf scalar values are number-like
    optional_null_paths = {
        'trace.result.best_by_total_paid',
        'result.best_by_total_paid',
    }

    def walk(obj, path=''):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                walk(item, f"{path}[{i}]")
        else:
            if obj is None:
                if path in optional_null_paths:
                    return
                raise ValueError(f'Non-numeric result at {path}: None')
            if isinstance(obj, str) and obj.strip() == '':
                raise ValueError(f'Empty string in numeric result at {path}')
            # allow booleans? treat as invalid for numeric outputs
            if isinstance(obj, bool):
                raise ValueError(f'Unexpected boolean in numeric result at {path}')
            # If scalar, ensure convertible to float when expected
            try:
                float(obj)
            except Exception:
                # allow non-numeric lists/dicts (they should have been deeper-walked)
                raise ValueError(f'Non-numeric scalar in result at {path}: {obj}')

    # Walk trace result and top-level result
    trace_res = solved.get('trace', {}).get('result', {})
    if trace_res:
        walk(trace_res, 'trace.result')
    else:
        # Fallback to checking top-level result
        walk(result, 'result')


def _validate_meta_output(meta_result: dict, contract: dict):
    """Basic validator for meta-analysis output against the knowledge base contract.

    This does superficial checks: presence of top-level keys and numeric types for key fields.
    Raises ValueError on mismatch.
    """
    if not isinstance(meta_result, dict):
        raise ValueError('Meta result is not a JSON object')

    required = contract.get('required_structure', {})
    # Check status
    status_values = required.get('status_values', [])
    status = meta_result.get('status')
    if status_values and status not in status_values:
        raise ValueError(f"Invalid or missing status: {status}")

    summary = meta_result.get('summary')
    if not isinstance(summary, dict):
        raise ValueError('Missing summary object')
    for key in ['income', 'expenses', 'savings_rate']:
        if key not in summary:
            raise ValueError(f"Summary missing '{key}'")
        if not _is_number_like(summary.get(key)):
            raise ValueError(f"Summary field '{key}' is not numeric")

    # category_analysis basic check
    cats = meta_result.get('category_analysis', [])
    if not isinstance(cats, list):
        raise ValueError('category_analysis must be a list')
    for i, c in enumerate(cats[:6]):
        if not isinstance(c, dict):
            raise ValueError(f'category_analysis[{i}] is not an object')
        if 'category' not in c or 'spent' not in c:
            raise ValueError(f'category_analysis[{i}] missing category or spent')
        if not _is_number_like(c.get('spent')):
            raise ValueError(f'category_analysis[{i}].spent is not numeric')



def _clear_meta_analysis_state():
    st.session_state.meta_analysis_result = None
    st.session_state.meta_analysis_metrics = None
    st.session_state.meta_analysis_error = None
    st.session_state.meta_analysis_source = None


def _clear_problem_solver_state():
    st.session_state.problem_draft = None
    st.session_state.problem_draft_text = ""
    st.session_state.problem_source_question = ""
    st.session_state.problem_solver_result = None
    st.session_state.problem_solver_narrative = None
    st.session_state.problem_solver_metrics = None
    st.session_state.problem_solver_error = None


def _infer_problem_type(payload: dict) -> str:
    """Infer a problem type from the JSON structure when the parser is incomplete."""
    ptype = (payload.get("problem_type") or "").lower().strip()
    if ptype:
        return ptype

    if payload.get("current_loan") and payload.get("proposed_loan"):
        return "refinance"
    if payload.get("loans"):
        return "compare_loans"
    if payload.get("structures"):
        return "debt_structures"
    if payload.get("cashflows"):
        return "npv_irr"
    if payload.get("nominal_return") is not None or payload.get("inflation") is not None:
        return "real_return"
    if payload.get("extra_payment") is not None:
        return "debt_payment_alternatives"
    if payload.get("future_value") is not None:
        return "present_value"
    if payload.get("present_value") is not None:
        return "future_value"
    if payload.get("principal") is not None and payload.get("periods") is not None:
        return "loan_payment"
    if payload.get("rate") is not None:
        return "rate_conversion"

    return ""


def _apply_financial_defaults(payload: dict) -> dict:
    """Inject explicit defaults so deterministic calculations can always proceed.

    IMPORTANT: This function must be a pure transformation of the authorized JSON.
    The same JSON input must always produce the same normalized payload.
    """
    normalized = copy.deepcopy(payload)
    normalized.setdefault("inputs", {})
    normalized.setdefault("defaults_used", [])
    normalized.setdefault("assumptions", [])
    normalized.setdefault("day_basis", 365)
    normalized.setdefault("periodicity", "monthly")
    normalized.setdefault("currency", "USD")

    def add_default(note: str):
        if note not in normalized["defaults_used"]:
            normalized["defaults_used"].append(note)

    def ensure_rate(container: dict, default_rate: float, note: str):
        if container.get("rate") in [None, "", 0, "0"]:
            container["rate"] = default_rate
            add_default(note)

    def _container_text(container: dict) -> str:
        return " ".join(str(v) for v in container.values()).lower()

    def _is_card_context(container: dict) -> bool:
        text = _container_text(container)
        return any(token in text for token in ["tarjeta", "credit card", "tc", "card"])

    def ensure_common_loan_fields(container: dict):
        card_hint = _is_card_context(container)
        ensure_rate(
            container,
            0.35 if card_hint else 0.10,
            "Tasa de tarjeta de crédito asumida: 35% anual" if card_hint else "Tasa financiera asumida: 10% anual",
        )
        if container.get("rate_type") in [None, ""]:
            container["rate_type"] = "effective_annual"
            add_default("Tipo de tasa asumido: efectiva anual")
        if container.get("periods") in [None, "", 0, "0"]:
            container["periods"] = 12
            add_default("Plazo asumido: 12 periodos mensuales")
        if container.get("method") in [None, ""]:
            container["method"] = "french"
            add_default("Método de amortización asumido: francés")

    normalized["problem_type"] = _infer_problem_type(normalized)
    ptype = normalized["problem_type"]

    if ptype in ["present_value", "future_value", "rate_conversion", "real_return", "npv_irr"]:
        if ptype == "present_value":
            normalized["inputs"].setdefault("future_value", normalized.get("future_value"))
            normalized["inputs"].setdefault("periods", normalized.get("periods"))
            normalized["inputs"].setdefault("rate", normalized.get("rate"))
        elif ptype == "future_value":
            normalized["inputs"].setdefault("present_value", normalized.get("present_value"))
            normalized["inputs"].setdefault("periods", normalized.get("periods"))
            normalized["inputs"].setdefault("rate", normalized.get("rate"))

        if ptype == "rate_conversion":
            normalized["inputs"].setdefault("rate", normalized.get("rate"))
            normalized["inputs"].setdefault("from_type", normalized.get("from_type", "effective_annual"))
        elif ptype == "real_return":
            normalized["inputs"].setdefault("nominal_return", normalized.get("nominal_return"))
            normalized["inputs"].setdefault("inflation", normalized.get("inflation"))
        elif ptype == "npv_irr":
            normalized["inputs"].setdefault("cashflows", normalized.get("cashflows", []))
            normalized["inputs"].setdefault("discount_rate", normalized.get("discount_rate"))

        if normalized["inputs"].get("rate") in [None, "", 0, "0"]:
            normalized["inputs"]["rate"] = 0.10
            add_default("Tasa financiera asumida: 10% anual")

        if ptype in ["present_value", "future_value"] and normalized["inputs"].get("periods") in [None, "", 0, "0"]:
            normalized["inputs"]["periods"] = 12
            add_default("Plazo asumido: 12 periodos")

        if ptype == "npv_irr" and normalized["inputs"].get("discount_rate") in [None, "", 0, "0"]:
            normalized["inputs"]["discount_rate"] = 0.10
            add_default("Tasa de descuento asumida: 10% anual")

        if ptype == "real_return":
            if normalized["inputs"].get("nominal_return") in [None, "", 0, "0"]:
                normalized["inputs"]["nominal_return"] = 0.10
                add_default("Rentabilidad nominal asumida: 10% anual")
            if normalized["inputs"].get("inflation") in [None, "", 0, "0"]:
                normalized["inputs"]["inflation"] = 0.04
                add_default("Inflación asumida: 4% anual")

    elif ptype == "loan_payment":
        normalized["inputs"].update({k: v for k, v in normalized.items() if k in ["principal", "rate", "rate_type", "periods", "method"]})
        ensure_common_loan_fields(normalized["inputs"])

    elif ptype == "compare_loans":
        loans = normalized["inputs"].get("loans") or normalized.get("loans") or []
        for loan in loans:
            ensure_common_loan_fields(loan)
        normalized["inputs"]["loans"] = loans

    elif ptype == "refinance":
        current_loan = normalized["inputs"].get("current_loan") or normalized.get("current_loan") or {}
        proposed_loan = normalized["inputs"].get("proposed_loan") or normalized.get("proposed_loan") or {}
        ensure_common_loan_fields(current_loan)
        ensure_common_loan_fields(proposed_loan)
        normalized["inputs"]["current_loan"] = current_loan
        normalized["inputs"]["proposed_loan"] = proposed_loan

    elif ptype == "debt_structures":
        structures = normalized["inputs"].get("structures") or normalized.get("structures") or []
        for structure in structures:
            ensure_common_loan_fields(structure)
        normalized["inputs"]["structures"] = structures

    elif ptype == "debt_payment_alternatives":
        normalized["inputs"].setdefault("principal", normalized.get("principal"))
        normalized["inputs"].setdefault("rate", normalized.get("rate"))
        normalized["inputs"].setdefault("rate_type", normalized.get("rate_type", "effective_annual"))
        normalized["inputs"].setdefault("periods", normalized.get("periods"))
        normalized["inputs"].setdefault("extra_payment", normalized.get("extra_payment", 0))
        ensure_common_loan_fields(normalized["inputs"])

    else:
        # Fallback: if the parser produced a partial loan-like payload, force a standard loan payment.
        normalized["problem_type"] = "loan_payment"
        normalized["inputs"].setdefault("principal", normalized.get("principal", normalized["inputs"].get("principal", 0)))
        normalized["inputs"].setdefault("rate", normalized.get("rate", normalized["inputs"].get("rate", 0.10)))
        normalized["inputs"].setdefault("rate_type", normalized.get("rate_type", "effective_annual"))
        normalized["inputs"].setdefault("periods", normalized.get("periods", 12))
        normalized["inputs"].setdefault("method", normalized.get("method", "french"))
        add_default("Tipo de problema inferido: loan_payment")
        ensure_common_loan_fields(normalized["inputs"])

    return normalized


def _interpret_problem_to_json(question: str, currency: str) -> dict:
    llm = st.session_state.analyzer.llm
    calls_before = llm.call_count
    cost_before = llm.total_cost

    system_prompt = """You are a financial math problem parser.
Convert a free-form user question into a normalized JSON payload for a deterministic calculator.

Supported problem_type values:
- present_value
- future_value
- loan_payment
- compare_loans
- rate_conversion
- refinance
- npv_irr
- real_return
- debt_structures
- debt_payment_alternatives

Rules:
- Return only valid JSON.
- If values (like interest rates, periods) are missing, you MUST inject reasonable numeric defaults directly into the inputs (e.g., 0.35 / 35% for a credit card rate, or 0.10 for personal loans, 0.08 for mortgages).
- You MUST document these injected assumptions specifically in the "defaults_used" array (e.g. "Tasa de tarjeta de crédito asumida: 35% anual").
- If ambiguity exists, explain it in ambiguity_disclosure.
- Use periodicity monthly and day_basis 365 unless user states otherwise.
- Prefer fields that can directly populate structured form boxes. For loan comparisons, return each loan as an object with name, principal, rate, rate_type, periods, and method.
- Do not omit fields that are already inferable from the question; if you must assume a default, keep it numeric and record it in defaults_used.

Required output shape:
{
  "problem_type": "...",
  "currency": "MXN|USD",
  "day_basis": 365,
  "periodicity": "monthly",
  "inputs": { ... },
  "assumptions": ["..."],
  "defaults_used": ["..."],
  "ambiguity_disclosure": "..."
}
"""

    user_prompt = f"""User question: {question}
Currency preference: {currency}
"""

    response = llm.make_call(user_prompt, system_prompt, expect_json=True)
    if not response:
        response = llm.make_call(user_prompt, system_prompt, expect_json=False)
    if not response:
        raise ValueError("No se pudo interpretar la pregunta.")

    parsed = _extract_json_response(response)

    calls_used = llm.call_count - calls_before
    cost_used = llm.total_cost - cost_before
    return {
        "draft": parsed,
        "metrics": {
            "llm_calls": max(calls_used, 0),
            "estimated_cost": max(cost_used, 0.0),
        },
    }


def _solve_authorized_problem(authorized_payload: dict) -> dict:
    deterministic_payload = _apply_financial_defaults(authorized_payload)
    solved = solve_problem(deterministic_payload)
    solved["context"] = {
        "currency": authorized_payload.get("currency", "USD"),
        "periodicity": authorized_payload.get("periodicity", "monthly"),
        "day_basis": authorized_payload.get("day_basis", 365),
        "defaults_used": deterministic_payload.get("defaults_used", []),
        "ambiguity_disclosure": authorized_payload.get("ambiguity_disclosure", ""),
    }
    solved["trace"]["assumptions"] = deterministic_payload.get("assumptions", []) + deterministic_payload.get("defaults_used", [])
    return solved


def _generate_problem_narrative(question: str, solved: dict) -> dict:
    llm = st.session_state.analyzer.llm
    calls_before = llm.call_count
    cost_before = llm.total_cost

    solver_type = (solved.get("type") or solved.get("result", {}).get("type") or "").lower()
    solved_result = solved.get("result", {}) or {}

    deterministic_summary = ""
    if solver_type == "compare_loans":
        comparisons = solved_result.get("comparison") or []
        winner = solved_result.get("best_by_total_paid") or {}
        if comparisons and isinstance(winner, dict):
            name = winner.get("name") or "la mejor opción"
            method = winner.get("method") or "desconocido"
            total_paid = winner.get("total_paid")
            total_interest = winner.get("total_interest")
            deterministic_summary = (
                f"Resultado principal: la opción con menor costo total es {name} "
                f"(método {method}), con pago total de {total_paid} y intereses totales de {total_interest}."
            )
        else:
            deterministic_summary = (
                "Resultado principal: el solver no recibió alternativas suficientes para comparar "
                "costos totales e intereses pagados."
            )

    system_prompt = """You are a financial analyst.
Write a concise but human-readable report in Spanish with this structure:
1) Interpretación del problema
2) Supuestos y defaults utilizados
3) Pasos de cálculo (claros)
4) Resultado principal
5) Interpretación ejecutiva
6) Recomendaciones accionables

Strict rules:
- Mention that this is educational and not regulated investment advice.
- Do not recommend specific financial products without user profile context.
- Always reference ONLY the deterministic results provided in the input (the solver's trace).
- Do NOT invent or introduce any new numeric values. Use only numbers present in the solver output (`trace.result` or top-level `result`).
- If a deterministic main result is provided in the user prompt, include it verbatim in the section '4) Resultado principal'.
- If you cannot produce the narrative without adding or guessing numbers, return an error instead of fabricating values.
"""

    user_prompt = f"""Original question:
{question}

Deterministic solution trace:
{json.dumps(solved, ensure_ascii=False, indent=2)}

Deterministic main result to include verbatim if available:
{deterministic_summary}
"""

    narrative = llm.make_call(user_prompt, system_prompt, expect_json=False)
    if not narrative:
        raise ValueError("No se pudo generar redacción ejecutiva.")

    if deterministic_summary and deterministic_summary not in narrative:
        narrative = f"{deterministic_summary}\n\n{narrative}"

    calls_used = llm.call_count - calls_before
    cost_used = llm.total_cost - cost_before
    return {
        "text": narrative,
        "metrics": {
            "llm_calls": max(calls_used, 0),
            "estimated_cost": max(cost_used, 0.0),
        },
    }


def _build_structured_problem_payload(problem_type: str, currency: str, values: dict) -> dict:
    """Build a deterministic, human-editable payload for direct calculation.

    The structured flow avoids free-form interpretation for common financial math
    patterns and produces a payload that can be reviewed before solving.
    """
    payload = {
        "problem_type": problem_type,
        "currency": currency,
        "day_basis": 365,
        "periodicity": "monthly",
        "inputs": {},
        "assumptions": [],
        "defaults_used": [],
        "ambiguity_disclosure": "",
    }

    if problem_type == "loan_payment":
        payload["inputs"] = {
            "principal": values.get("principal", 0),
            "rate": values.get("rate", 0),
            "rate_type": values.get("rate_type", "effective_annual"),
            "periods": values.get("periods", 0),
            "method": values.get("method", "french"),
        }
    elif problem_type == "compare_loans":
        payload["inputs"] = {
            "loans": values.get("loans", []),
        }
    elif problem_type == "present_value":
        payload["inputs"] = {
            "future_value": values.get("future_value", 0),
            "rate": values.get("rate", 0),
            "periods": values.get("periods", 0),
        }
    elif problem_type == "future_value":
        payload["inputs"] = {
            "present_value": values.get("present_value", 0),
            "rate": values.get("rate", 0),
            "periods": values.get("periods", 0),
        }
    elif problem_type == "rate_conversion":
        payload["inputs"] = {
            "rate": values.get("rate", 0),
            "from_type": values.get("from_type", "effective_annual"),
        }
    elif problem_type == "npv_irr":
        payload["inputs"] = {
            "cashflows": values.get("cashflows", []),
            "discount_rate": values.get("discount_rate", 0),
        }
    elif problem_type == "real_return":
        payload["inputs"] = {
            "nominal_return": values.get("nominal_return", 0),
            "inflation": values.get("inflation", 0),
        }
    else:
        payload["inputs"] = values

    return payload


def _is_valid_structured_payload(payload: dict) -> tuple[bool, str]:
    """Validate the structured payload before allowing resolution."""
    ptype = (payload.get("problem_type") or "").lower().strip()
    inputs = payload.get("inputs", {}) or {}

    def has_positive_number(value):
        try:
            return float(value) > 0
        except Exception:
            return False

    if ptype == "loan_payment":
        if not has_positive_number(inputs.get("principal")):
            return False, "Ingresa un principal mayor que cero."
        if not has_positive_number(inputs.get("rate")):
            return False, "Ingresa una tasa mayor que cero."
        if not has_positive_number(inputs.get("periods")):
            return False, "Ingresa un plazo mayor que cero."
        return True, ""

    if ptype == "compare_loans":
        loans = inputs.get("loans") or []
        if len(loans) < 2:
            return False, "Necesitas al menos dos préstamos para comparar."
        for index, loan in enumerate(loans, start=1):
            if not has_positive_number(loan.get("principal")):
                return False, f"El préstamo {index} necesita un principal mayor que cero."
            if not has_positive_number(loan.get("rate")):
                return False, f"El préstamo {index} necesita una tasa mayor que cero."
            if not has_positive_number(loan.get("periods")):
                return False, f"El préstamo {index} necesita un plazo mayor que cero."
        return True, ""

    if ptype == "present_value":
        if not has_positive_number(inputs.get("future_value")):
            return False, "Ingresa un valor futuro mayor que cero."
        if not has_positive_number(inputs.get("rate")):
            return False, "Ingresa una tasa mayor que cero."
        if not has_positive_number(inputs.get("periods")):
            return False, "Ingresa un número de periodos mayor que cero."
        return True, ""

    if ptype == "future_value":
        if not has_positive_number(inputs.get("present_value")):
            return False, "Ingresa un valor presente mayor que cero."
        if not has_positive_number(inputs.get("rate")):
            return False, "Ingresa una tasa mayor que cero."
        if not has_positive_number(inputs.get("periods")):
            return False, "Ingresa un número de periodos mayor que cero."
        return True, ""

    if ptype == "rate_conversion":
        if not has_positive_number(inputs.get("rate")):
            return False, "Ingresa una tasa mayor que cero."
        return True, ""

    if ptype == "npv_irr":
        cashflows = inputs.get("cashflows") or []
        if len(cashflows) < 2:
            return False, "Ingresa al menos dos flujos de caja."
        return True, ""

    if ptype == "real_return":
        if not has_positive_number(inputs.get("nominal_return")):
            return False, "Ingresa una rentabilidad nominal mayor que cero."
        return True, ""

    return False, "Tipo de problema no soportado por el formulario estructurado."


def _calculate_quick_loan_comparison(loans: list) -> dict:
    """Calculate comparison metrics for loans without full solver overhead."""
    from utils.financial_solver import loan_payment
    
    results = []
    for loan in loans:
        # Allow 0% interest rate, but require principal and periods
        principal = loan.get("principal")
        rate = loan.get("rate")
        periods = loan.get("periods")
        if not (principal and periods and rate is not None):
            continue
        try:
            calc = loan_payment(loan)
            results.append({
                "name": loan.get("name", "Préstamo"),
                "principal": float(loan.get("principal", 0)),
                "rate": float(loan.get("rate", 0)),
                "periods": int(loan.get("periods", 0)),
                "estimated_payment": calc.get("result", {}).get("estimated_payment", 0),
                "total_paid": calc.get("result", {}).get("total_paid", 0),
                "total_interest": calc.get("result", {}).get("total_interest", 0),
            })
        except Exception:
            pass
    
    if results:
        results_sorted = sorted(results, key=lambda x: x.get("total_paid", 0))
        for i, r in enumerate(results_sorted, 1):
            r["rank"] = i
    
    return {"results": results}


def _prime_structured_form_state(draft: dict):
    """Populate widget state from a parsed natural-language draft."""
    if not isinstance(draft, dict):
        return

    ptype = (draft.get("problem_type") or "").lower().strip()
    inputs = draft.get("inputs", {}) or {}

    st.session_state["structured_problem_type"] = ptype or "loan_payment"

    if ptype == "loan_payment":
        st.session_state["lp_principal"] = inputs.get("principal", st.session_state.get("lp_principal", 0.0))
        st.session_state["lp_rate"] = inputs.get("rate", st.session_state.get("lp_rate", 0.0))
        st.session_state["lp_periods"] = inputs.get("periods", st.session_state.get("lp_periods", 0))
        st.session_state["lp_rate_type"] = inputs.get("rate_type", st.session_state.get("lp_rate_type", "effective_annual"))
        st.session_state["lp_method"] = inputs.get("method", st.session_state.get("lp_method", "french"))

    elif ptype == "compare_loans":
        loans = inputs.get("loans", []) or []
        st.session_state["cl_loan_count"] = max(len(loans) + 1, 2)
        for index, loan in enumerate(loans, start=1):
            st.session_state[f"cl_name_{index}"] = loan.get("name", st.session_state.get(f"cl_name_{index}", f"Opción {index}"))
            st.session_state[f"cl_principal_{index}"] = loan.get("principal", st.session_state.get(f"cl_principal_{index}", 0.0))
            st.session_state[f"cl_rate_{index}"] = loan.get("rate", st.session_state.get(f"cl_rate_{index}", 0.0))
            st.session_state[f"cl_periods_{index}"] = loan.get("periods", st.session_state.get(f"cl_periods_{index}", 0))
            st.session_state[f"cl_rate_type_{index}"] = loan.get("rate_type", st.session_state.get(f"cl_rate_type_{index}", "effective_annual"))
            st.session_state[f"cl_method_{index}"] = loan.get("method", st.session_state.get(f"cl_method_{index}", "french"))

    elif ptype == "present_value":
        st.session_state["pv_future"] = inputs.get("future_value", st.session_state.get("pv_future", 0.0))
        st.session_state["pv_rate"] = inputs.get("rate", st.session_state.get("pv_rate", 0.0))
        st.session_state["pv_periods"] = inputs.get("periods", st.session_state.get("pv_periods", 0))

    elif ptype == "future_value":
        st.session_state["fv_present"] = inputs.get("present_value", st.session_state.get("fv_present", 0.0))
        st.session_state["fv_rate"] = inputs.get("rate", st.session_state.get("fv_rate", 0.0))
        st.session_state["fv_periods"] = inputs.get("periods", st.session_state.get("fv_periods", 0))

    elif ptype == "rate_conversion":
        st.session_state["rc_rate"] = inputs.get("rate", st.session_state.get("rc_rate", 0.0))
        st.session_state["rc_from_type"] = inputs.get("from_type", st.session_state.get("rc_from_type", "effective_annual"))

    elif ptype == "npv_irr":
        cashflows = inputs.get("cashflows", []) or []
        st.session_state["npv_cashflows"] = ", ".join(str(x) for x in cashflows)
        st.session_state["npv_discount_rate"] = inputs.get("discount_rate", st.session_state.get("npv_discount_rate", 0.0))

    elif ptype == "real_return":
        st.session_state["rr_nominal"] = inputs.get("nominal_return", st.session_state.get("rr_nominal", 0.0))
        st.session_state["rr_inflation"] = inputs.get("inflation", st.session_state.get("rr_inflation", 0.0))


def _render_structured_problem_builder(currency: str):
    """Render a structured calculator builder and return a payload if submitted."""
    st.subheader("🧩 Caja de cálculo estructurada")
    st.caption("Escribe el problema en lenguaje natural; el LLM llenará las cajas y luego tú solo revisas antes de autorizar.")
    
    # Add custom CSS for orange button
    st.markdown("""
        <style>
            [data-testid="column"] button[kind="primary"] {
                background-color: #FF9800 !important;
                color: white !important;
                padding: 12px 24px !important;
                font-size: 16px !important;
                font-weight: 600 !important;
                border-radius: 8px !important;
                height: 48px !important;
            }
            [data-testid="column"] button[kind="primary"]:hover {
                background-color: #F57C00 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    structured_question = st.text_area(
        "Describe el problema en lenguaje natural",
        value=st.session_state.get("structured_natural_question", ""),
        height=110,
        placeholder="Ej: Tengo dos créditos, uno al 32% y otro al 24%, ambos a 24 meses. Quiero saber cuál me conviene más.",
        key="structured_natural_question",
    )

    # Large orange button for interpretation
    interpret_clicked = st.button(
        "🤖 Interpretar y llenar cajas",
        type="primary",
        use_container_width=True,
        help="El LLM interpretará tu problema y rellenará automáticamente las cajas de cálculo"
    )
    
    if interpret_clicked:
        if not structured_question.strip():
            st.warning("Escribe un problema en lenguaje natural antes de interpretar.")
        else:
            try:
                with st.spinner("Traduciendo el problema a campos estructurados..."):
                    parsed = _interpret_problem_to_json(structured_question.strip(), currency)
                    draft = parsed["draft"]
                    _prime_structured_form_state(draft)
                    st.session_state.problem_source_question = structured_question.strip()
                    st.session_state.problem_draft = draft
                    st.session_state.problem_draft_text = json.dumps(draft, ensure_ascii=False, indent=2)
                    st.session_state.problem_solver_metrics = {
                        "parse_llm_calls": parsed["metrics"]["llm_calls"],
                        "parse_cost": parsed["metrics"]["estimated_cost"],
                        "narrative_llm_calls": 0,
                        "narrative_cost": 0.0,
                    }
                    st.session_state.problem_solver_error = None
                    st.success("Cajas llenas. Revisa los datos y autoriza el cálculo.")
            except Exception as e:
                st.session_state.problem_solver_error = str(e)

    problem_type = st.selectbox(
        "Tipo de cálculo",
        ["loan_payment", "compare_loans", "present_value", "future_value", "rate_conversion", "npv_irr", "real_return"],
        index=0,
        format_func=lambda x: {
            "loan_payment": "Pago de préstamo",
            "compare_loans": "Comparar préstamos",
            "present_value": "Valor presente",
            "future_value": "Valor futuro",
            "rate_conversion": "Conversión de tasa",
            "npv_irr": "VAN / TIR",
            "real_return": "Rentabilidad real",
        }.get(x, x),
        key="structured_problem_type",
    )

    # Special handling for compare_loans (outside form due to buttons)
    if problem_type == "compare_loans":
        st.markdown("Define los préstamos que deseas comparar. Siempre habrá una caja vacía para agregar más.")
        
        # Initialize loan count if needed
        if "cl_loan_count" not in st.session_state:
            st.session_state["cl_loan_count"] = 2
        
        # Add more loans button (outside form)
        col_add, col_calc = st.columns([1, 2])
        with col_add:
            if st.button("➕ Agregar otra opción", use_container_width=True, key="cl_add_btn"):
                st.session_state["cl_loan_count"] += 1
                st.rerun()
        
        # Build loans data from session state
        loans = []
        num_loans = st.session_state.get("cl_loan_count", 2)
        
        for idx in range(1, num_loans + 1):
            with st.expander(f"Préstamo {idx}", expanded=(idx <= 2)):
                loan_col1, loan_col2, loan_col3 = st.columns(3)
                with loan_col1:
                    name = st.text_input(f"Nombre {idx}", value=st.session_state.get(f"cl_name_{idx}", f"Opción {idx}"), key=f"cl_name_{idx}")
                    principal = st.number_input(f"Principal {idx}", min_value=0.0, value=float(st.session_state.get(f"cl_principal_{idx}", 0.0)), step=1000.0, key=f"cl_principal_{idx}")
                with loan_col2:
                    rate = st.number_input(f"Tasa anual {idx}", min_value=0.0, value=float(st.session_state.get(f"cl_rate_{idx}", 0.0)), step=0.01, format="%.4f", key=f"cl_rate_{idx}")
                    periods = st.number_input(f"Plazo {idx}", min_value=0, value=int(st.session_state.get(f"cl_periods_{idx}", 0)), step=1, key=f"cl_periods_{idx}")
                with loan_col3:
                    rate_type = st.selectbox(f"Tipo de tasa {idx}", ["effective_annual", "nominal_annual", "effective_monthly"], index=0, key=f"cl_rate_type_{idx}")
                    method = st.selectbox(f"Método {idx}", ["french", "german", "american"], index=0, key=f"cl_method_{idx}")
                
                loans.append({
                    "name": name,
                    "principal": principal,
                    "rate": rate,
                    "rate_type": rate_type,
                    "periods": periods,
                    "method": method,
                })
        
        # Quick comparison button and results
        st.divider()
        if st.button("📊 Calcular comparación rápida", type="secondary", use_container_width=True, key="cl_calc_btn"):
            # Allow 0% rate; just check principal and periods
            valid_loans = [l for l in loans if l["principal"] > 0 and l["periods"] > 0 and l["rate"] is not None]
            if len(valid_loans) >= 2:
                comparison_result = _calculate_quick_loan_comparison(loans)
                if comparison_result.get("results"):
                    st.subheader("📊 Resultados de comparación")
                    
                    # Table view
                    df_comparison = pd.DataFrame([
                        {
                            "🏆": "🥇" if r["rank"] == 1 else "🥈" if r["rank"] == 2 else "🥉",
                            "Opción": r["name"],
                            "Principal": f"${r['principal']:,.2f}",
                            "Tasa": f"{r['rate']*100:.2f}%",
                            "Plazo": f"{r['periods']} meses",
                            "Cuota": f"${r['estimated_payment']:,.2f}",
                            "Total Pagado": f"${r['total_paid']:,.2f}",
                            "Intereses": f"${r['total_interest']:,.2f}",
                        }
                        for r in comparison_result["results"]
                    ])
                    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
                    
                    # Winner info
                    winner = comparison_result["results"][0] if comparison_result["results"] else None
                    if winner:
                        st.success(f"✅ **Mejor opción:** {winner['name']} con pago total de ${winner['total_paid']:,.2f}")
            else:
                st.warning("Necesitas al menos 2 préstamos completamente llenos (principal, plazo > 0; tasa >= 0).")
        st.divider()
        
        # For compare_loans, just show results - no need for JSON editor or solver
        return None

    # For all other types, use the form
    payload = None
    with st.form("structured_problem_form", clear_on_submit=False):
        values = {}

        if problem_type == "loan_payment":
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                values["principal"] = st.number_input("Principal", min_value=0.0, value=0.0, step=1000.0, key="lp_principal")
            with col_b:
                values["rate"] = st.number_input("Tasa anual", min_value=0.0, value=0.0, step=0.01, format="%.4f", key="lp_rate")
            with col_c:
                values["periods"] = st.number_input("Plazo (periodos)", min_value=0, value=0, step=1, key="lp_periods")
            col_d, col_e = st.columns(2)
            with col_d:
                values["rate_type"] = st.selectbox("Tipo de tasa", ["effective_annual", "nominal_annual", "effective_monthly"], index=0, key="lp_rate_type")
            with col_e:
                values["method"] = st.selectbox("Método", ["french", "german", "american"], index=0, key="lp_method")

        elif problem_type == "present_value":
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                values["future_value"] = st.number_input("Valor futuro", min_value=0.0, value=0.0, step=1000.0, key="pv_future")
            with col_b:
                values["rate"] = st.number_input("Tasa anual", min_value=0.0, value=0.0, step=0.01, format="%.4f", key="pv_rate")
            with col_c:
                values["periods"] = st.number_input("Plazo", min_value=0, value=0, step=1, key="pv_periods")

        elif problem_type == "future_value":
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                values["present_value"] = st.number_input("Valor presente", min_value=0.0, value=0.0, step=1000.0, key="fv_present")
            with col_b:
                values["rate"] = st.number_input("Tasa anual", min_value=0.0, value=0.0, step=0.01, format="%.4f", key="fv_rate")
            with col_c:
                values["periods"] = st.number_input("Plazo", min_value=0, value=0, step=1, key="fv_periods")

        elif problem_type == "rate_conversion":
            col_a, col_b = st.columns(2)
            with col_a:
                values["rate"] = st.number_input("Tasa", min_value=0.0, value=0.0, step=0.01, format="%.4f", key="rc_rate")
            with col_b:
                values["from_type"] = st.selectbox("Convertir desde", ["effective_annual", "nominal_annual", "effective_monthly"], index=0, key="rc_from_type")

        elif problem_type == "npv_irr":
            st.markdown("Usa una lista simple de flujos separada por comas. Ejemplo: -1000, 300, 400, 500")
            cashflows_text = st.text_input("Flujos de caja", value="", key="npv_cashflows")
            values["discount_rate"] = st.number_input("Tasa de descuento", min_value=0.0, value=0.0, step=0.01, format="%.4f", key="npv_discount_rate")
            values["cashflows"] = [float(x.strip()) for x in cashflows_text.split(",") if x.strip()] if cashflows_text.strip() else []

        elif problem_type == "real_return":
            col_a, col_b = st.columns(2)
            with col_a:
                values["nominal_return"] = st.number_input("Rentabilidad nominal", min_value=0.0, value=0.0, step=0.01, format="%.4f", key="rr_nominal")
            with col_b:
                values["inflation"] = st.number_input("Inflación", min_value=0.0, value=0.0, step=0.01, format="%.4f", key="rr_inflation")

        submit = st.form_submit_button("📦 Crear caja de cálculo", use_container_width=True)

    if not submit:
        return None

    payload = _build_structured_problem_payload(problem_type, currency, values)
    valid, message = _is_valid_structured_payload(payload)
    if not valid:
        st.warning(message)
        return None

    return payload


def render_problem_solver_page():
    render_problem_solver_page_v2()


def _generate_executive_summary(result: dict, income: float, expenses: float, savings: float, savings_rate: float, health_status: str) -> str:
    """Generate a professional narrative executive summary.

    This version escapes Markdown-sensitive characters in inserted text and formats
    numeric values consistently to avoid unintended font/format changes in Streamlit.
    """

    import html

    def esc(text: str) -> str:
        if text is None:
            return ""
        return html.escape(str(text), quote=False)

    categories = result.get('category_analysis', []) or []

    # Calculate category totals (robust to missing data)
    fixed_categories = ['bills_utilities', 'fees']
    variable_categories = ['groceries', 'transportation']

    total_fixed = 0.0
    total_variable = 0.0
    total_discretionary = 0.0

    for cat in categories:
        cat_name = cat.get('category', '')
        spent = float(cat.get('spent') or 0)
        if cat_name in fixed_categories:
            total_fixed += spent
        elif cat_name in variable_categories:
            total_variable += spent
        else:
            total_discretionary += spent

    # Safe formatting helpers
    def fmt_money(x):
        try:
            return f"${float(x):,.2f}"
        except Exception:
            return f"{x}"

    def pct(part, whole):
        try:
            return (float(part) / float(whole) * 100) if float(whole) != 0 else 0.0
        except Exception:
            return 0.0

    fixed_pct = pct(total_fixed, expenses)
    variable_pct = pct(total_variable, expenses)
    discretionary_pct = pct(total_discretionary, expenses)

    income_s = fmt_money(income)
    expenses_s = fmt_money(expenses)
    savings_s = fmt_money(savings)
    fixed_s = fmt_money(total_fixed)
    variable_s = fmt_money(total_variable)
    discretionary_s = fmt_money(total_discretionary)

    # Build narrative as controlled HTML to avoid Markdown formatting side effects
    if health_status == "good":
        opening = (f"<p><strong>Situación Financiera General:</strong> Tu perfil financiero muestra una posición sólida y estable. "
                   f"Con ingresos mensuales de {income_s} y gastos de {expenses_s}, logras un ahorro neto de {savings_s} "
                   f"(equivalente al {savings_rate*100:.1f}% de tus ingresos).</p>")
    elif health_status == "risk":
        opening = (f"<p><strong>Situación Financiera General:</strong> Tu perfil financiero requiere atención inmediata. "
                   f"Actualmente gastas {expenses_s} de cada {income_s} que ganas, dejando tan solo {savings_s} de ahorro mensual ({savings_rate*100:.1f}%). "
                   f"Existen oportunidades claras de optimización que deben abordarse con urgencia.</p>")
    else:
        opening = (f"<p><strong>Situación Financiera General:</strong> Tu perfil financiero es neutral. "
                   f"Con ingresos de {income_s} y gastos de {expenses_s}, logras un ahorro de {savings_s} ({savings_rate*100:.1f}%). "
                   f"Hay margen para mejora en varios aspectos de tu gestión financiera.</p>")

    breakdown = (f"<p><strong>Estructura de Gastos:</strong> Tu gasto total se distribuye en tres categorías clave. "
                 f"Los gastos fijos representan {fixed_pct:.0f}% del presupuesto ({fixed_s}). "
                 f"Los gastos variables representan {variable_pct:.0f}% ({variable_s}). "
                 f"Los gastos discrecionales alcanzan {discretionary_pct:.0f}% ({discretionary_s}).</p>")

    if discretionary_pct > 25:
        finding = (f"<p><strong>Hallazgo Principal:</strong> Se identifica un nivel significativo de gasto discrecional ({discretionary_pct:.0f}%). "
                   f"Reducir esta categoría hacia 15-20% podría liberar aproximadamente {fmt_money(total_discretionary * 0.40)} mensuales para ahorros e inversión.</p>")
    elif discretionary_pct > 15:
        finding = (f"<p><strong>Hallazgo Principal:</strong> El gasto discrecional está en un nivel moderado ({discretionary_pct:.0f}%). "
                   f"Pequeños ajustes podrían liberar cerca de {fmt_money(total_discretionary * 0.25)} mensuales.</p>")
    else:
        finding = (f"<p><strong>Hallazgo Principal:</strong> El gasto discrecional está bien controlado en {discretionary_pct:.0f}%. "
                   f"Mantienes una disciplina sólida en áreas de discreción.</p>")

    recommendations = ("<p><strong>Recomendaciones Estratégicas:</strong> Basado en este análisis, se sugieren las siguientes acciones: "
                       "(1) Revisar las suscripciones y gastos recurrentes; "
                       "(2) Implementar un presupuesto categorizado adaptado a tu perfil; "
                       "(3) Automatizar transferencias a ahorro justo al recibir ingresos.</p>")

    next_steps = ("<p><strong>Próximos Pasos:</strong> Consulta la sección de Ratios Financieros para contexto, revisa los Problemas Detectados si existen, "
                  "y considera implementar las acciones listadas en Recomendaciones. Genera este reporte mensualmente para monitorear progreso.</p>")

    # Assemble paragraphs (escape variable fragments if needed). We escape only
    # values derived from user data; the template's Markdown (bold headings)
    # remains intact so headings render correctly.
    paragraphs = [opening, breakdown, finding, recommendations, next_steps]
    return "".join(paragraphs)


def _render_meta_analysis_result(meta_state: dict):
    st.subheader("🧠 Meta Analysis Report")
    metrics = meta_state['metrics']
    result = meta_state['result']
    
    # System metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Meta LLM Calls", f"{metrics['llm_calls']}")
    with metric_col2:
        st.metric("Meta Cost", f"${metrics['estimated_cost']:.4f}")
    with metric_col3:
        st.metric("Total Cost", f"${metrics['total_cost']:.4f}")
    
    st.divider()
    
    summary = result.get('summary', {})
    income = summary.get('income', 0)
    expenses = summary.get('expenses', 0)
    savings = income - expenses
    savings_rate = summary.get('savings_rate', 0)
    health_status = summary.get('financial_health', 'neutral')
    
    # 1. Resumen General
    st.markdown("## 📊 1. Resumen General")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Ingresos", f"${income:.2f}")
    with col2:
        st.metric("💸 Gastos", f"${expenses:.2f}")
    with col3:
        st.metric("💳 Ahorro Neto", f"${savings:.2f}")
    with col4:
        health_emoji = {"good": "✔️", "risk": "⚠️", "critical": "🚨", "neutral": "➖"}.get(health_status, "➖")
        st.metric(f"{health_emoji} Estado", health_status.upper())
    
    st.write(f"**Tasa de ahorro: {savings_rate*100:.1f}%**")
    if savings_rate > 0.20:
        st.success("✔️ Excelente: ahorras más de lo que gastas.")
    elif savings_rate > 0.10:
        st.info("✔️ Bueno: estás ahorrando regularmente.")
    else:
        st.warning("⚠️ Atención: necesitas aumentar tu tasa de ahorro.")
    
    st.divider()
    
    # 2. Análisis de Categorías
    st.markdown("## 📋 2. Análisis de Gastos por Categoría")
    categories = result.get('category_analysis', [])
    
    if categories:
        # Separar en fijos, variables y discrecionales (aproximación)
        fixed_categories = ['bills_utilities', 'fees']
        variable_categories = ['groceries', 'transportation']
        discretionary_categories = ['food_dining', 'shopping', 'entertainment']
        
        total_fixed = 0
        total_variable = 0
        total_discretionary = 0
        
        for cat in categories:
            cat_name = cat.get('category', '')
            spent = cat.get('spent', 0)
            
            if cat_name in fixed_categories:
                total_fixed += spent
            elif cat_name in variable_categories:
                total_variable += spent
            else:
                total_discretionary += spent
        
        # Mostrar desglose
        st.markdown("### 🔴 Gastos Fijos (Obligatorios)")
        fixed_items = [c for c in categories if c.get('category', '') in fixed_categories]
        for item in fixed_items:
            st.write(f"• **{item.get('category', '').replace('_', ' ').title()}**: ${item.get('spent', 0):.2f}")
        st.write(f"**Total fijos:** ${total_fixed:.2f} ({total_fixed/expenses*100:.0f}% de gastos)")
        
        st.markdown("### 🟡 Gastos Variables (Necesarios)")
        variable_items = [c for c in categories if c.get('category', '') in variable_categories]
        for item in variable_items:
            st.write(f"• **{item.get('category', '').replace('_', ' ').title()}**: ${item.get('spent', 0):.2f}")
        st.write(f"**Total variables:** ${total_variable:.2f} ({total_variable/expenses*100:.0f}% de gastos)")
        
        st.markdown("### 🔵 Gastos Discrecionales (Optimizable)")
        discretionary_items = [c for c in categories if c.get('category', '') not in fixed_categories + variable_categories]
        for item in discretionary_items:
            st.write(f"• **{item.get('category', '').replace('_', ' ').title()}**: ${item.get('spent', 0):.2f}")
        st.write(f"**Total discrecional:** ${total_discretionary:.2f} ({total_discretionary/expenses*100:.0f}% de gastos)")
        
        if total_discretionary / expenses > 0.25:
            st.warning("⚠️ **Alto gasto discrecional**: Aquí está el principal potencial de optimización.")
    
    st.divider()
    
    # 3. Ratios Financieros Clave
    st.markdown("## 🧮 3. Ratios Financieros Clave")
    ratios = result.get('ratios', {})
    
    if ratios:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'savings_rate' in ratios:
                sr = ratios['savings_rate'].get('value', 0)
                st.write(f"**Tasa de Ahorro:** {sr*100:.1f}%")
                if sr > 0.20:
                    st.success("✔️ Muy buena (ideal >20%)")
                elif sr > 0.10:
                    st.info("✔️ Aceptable")
                else:
                    st.warning("⚠️ Por debajo del ideal")
            
            if 'housing_ratio' in ratios:
                hr = ratios['housing_ratio'].get('value', 0)
                st.write(f"**Carga de Vivienda:** {hr*100:.1f}%")
                if hr < 0.30:
                    st.success("✔️ Saludable (<30%)")
                else:
                    st.warning("⚠️ Elevado (>30%)")
        
        with col2:
            if 'debt_ratio' in ratios:
                dr = ratios['debt_ratio'].get('value', 0)
                st.write(f"**Ratio Deuda/Ingreso:** {dr*100:.1f}%")
                if dr < 0.15:
                    st.success("✔️ Bajo → buena capacidad financiera")
                else:
                    st.warning("⚠️ Moderado a alto")
            
            if 'discretionary_ratio' in ratios:
                disr = ratios['discretionary_ratio'].get('value', 0)
                st.write(f"**Gasto Discrecional:** {disr*100:.1f}%")
                if disr < 0.20:
                    st.success("✔️ Controlado")
                else:
                    st.warning("⚠️ Elevado - oportunidad de optimización")
    
    st.divider()
    
    # 4. Problemas Detectados
    detections = result.get('detections', [])
    st.markdown("## ⚠️ 4. Problemas Detectados")
    if detections:
        for detection in detections:
            msg = detection.get('message', detection.get('type', 'Detection'))
            if msg and msg.strip():
                st.write(f"• {msg}")
    else:
        st.info("✔️ No se detectaron problemas significativos en este análisis.")
    
    st.divider()
    
    # 5. Recomendaciones Estratégicas
    st.markdown("## 💡 5. Recomendaciones Estratégicas")
    recommendations = result.get('recommendations', [])
    
    if recommendations:
        for idx, rec in enumerate(recommendations, 1):
            priority = rec.get('priority', 'medium').upper()
            priority_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(priority, "➖")
            playbook_title = rec.get('playbook_id', 'Recomendación').replace('_', ' ').title()
            
            st.markdown(f"### {priority_emoji} Acción {idx}: {playbook_title}")
            st.write(f"**Impacto estimado:** {rec.get('impact_estimate', 'N/A')}")
            
            actions = rec.get('actions', [])
            if actions:
                st.write("**Pasos a seguir:**")
                for action in actions:
                    st.write(f"  • {action}")
    else:
        st.info("No hay recomendaciones específicas en este momento.")
    
    st.divider()
    
    # 6. Conclusión
    st.markdown("## 📈 6. Conclusión")
    
    if health_status == "good":
        st.success(f"✔️ **Tu situación financiera es SALUDABLE**\n\n"
                   f"Ingresos: ${income:.2f} | Gastos: ${expenses:.2f} | Ahorro: ${savings:.2f}\n\n"
                   f"Estás en una buena posición. Mantén la disciplina y considera optimizar gastos discrecionales para acelerar tu acumulación de riqueza.")
    elif health_status == "risk":
        st.warning(f"⚠️ **Tu situación financiera requiere ATENCIÓN**\n\n"
                   f"Ingresos: ${income:.2f} | Gastos: ${expenses:.2f} | Ahorro: ${savings:.2f}\n\n"
                   f"Existen áreas de riesgo. Prioriza reducir gastos discrecionales e implementar las recomendaciones.")
    else:
        st.info(f"➖ **Tu situación financiera es NEUTRAL**\n\n"
                f"Ingresos: ${income:.2f} | Gastos: ${expenses:.2f} | Ahorro: ${savings:.2f}\n\n"
                f"Hay oportunidades de mejora. Revisa las recomendaciones y toma acción.")
    
    st.divider()
    
    # Data Quality
    data_quality = result.get('data_quality', {})
    confidence = data_quality.get('confidence', 'unknown').upper()
    
    st.caption(f"✓ **Confianza de datos:** {confidence} | "
               f"⚠️ **Importante**: Este análisis debe ser revisado por un experto financiero antes de actuar.")
    
    st.divider()
    
    # Resumen Ejecutivo - Narrativa Profesional
    st.markdown("## 📄 Resumen Ejecutivo")
    
    executive_summary = _generate_executive_summary(result, income, expenses, savings, savings_rate, health_status)
    st.markdown(executive_summary, unsafe_allow_html=True)
    
    st.divider()
    
    st.caption("*Este reporte fue generado automáticamente por el sistema de asesoría financiera. "
               "Para cambios en tus estrategias de inversión o deudas, consulta siempre con un asesor financiero profesional.*")

if hasattr(st, 'dialog'):
    @st.dialog("Category Items")
    def _show_category_items_modal(category_label: str, transactions: list):
        st.subheader(f"{category_label} - Items")
        items = _get_category_items_for_modal(transactions, category_label)

        if not items:
            st.info("No spending items found for this category.")
            return

        for item in items:
            amount_formatted = f"${item['Amount']:.2f}"
            st.write(f"{item['Description']} | {amount_formatted}")
else:
    def _show_category_items_modal(category_label: str, transactions: list):
        st.warning("Your Streamlit version does not support modal dialogs. Please upgrade Streamlit to use this feature.")

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'analyzer' not in st.session_state:
        load_dotenv()
        api_key = _get_openai_api_key()
        
        if api_key:
            with st.spinner("🏗️ Initializing 3-agent system..."):
                st.session_state.analyzer = BankStatementAnalyzer(api_key)
            st.success("✅ System ready!")
        else:
            st.session_state.analyzer = None
            st.error("❌ Missing OPENAI_API_KEY. In Streamlit Cloud, add it in App secrets; locally, set it in .env.")

    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None

    if 'generate_ai_insights' not in st.session_state:
        st.session_state.generate_ai_insights = False

    if 'meta_analysis_result' not in st.session_state:
        _clear_meta_analysis_state()

    if 'problem_draft' not in st.session_state:
        _clear_problem_solver_state()

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Bank Statement Analyzer",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session
    initialize_session_state()
    
    # Main header
    st.title("🏦 Bank Statement Analyzer")
    st.subheader("Hybrid Multi-Agent Financial Analysis System")
    
    # Educational warning
    st.warning("⚠️ **Educational Project Only** - Do not use with real financial data containing sensitive information")
    
    # Sidebar - System Information
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:12px; margin:0.2rem 0 1rem 0;">
                <div style="font-size:2.2rem; font-weight:900; line-height:1; letter-spacing:-0.06em; color:#203A6B;">
                    h<span style="color:#F0B429;">w</span><span style="color:#2CA36A;">a</span>
                </div>
                <div style="line-height:1.05;">
                    <div style="font-size:1.15rem; font-weight:700; color:#203A6B;">Hispanic Wealth</div>
                    <div style="font-size:1.1rem; font-style:italic; color:#2CA36A;">Advisors<sup style="font-size:0.55em; vertical-align:super;">SM</sup></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        workspace_mode = st.radio(
            "Módulo",
            ["Análisis de cartola", "Problemas cotidianos"],
            index=0,
            help="Selecciona entre análisis de estados de cuenta y resolución de problemas financieros cotidianos"
        )
        
        if st.session_state.analyzer:
            st.success("✅ All agents initialized")
            
            # Agent information
            st.subheader("Agent Architecture")
            st.write("🏗️ **Agent 1**: Document Processor")
            st.caption("Extracts & parses PDF (0 LLM calls)")
            
            st.write("🧠 **Agent 2**: Content Analyzer") 
            st.caption("Smart categorization (1 LLM call)")
            
            st.write("📊 **Agent 3**: Analysis Generator")
            st.caption("Financial insights (0-1 LLM calls)")
            
            st.divider()
            
            # Project info
            st.subheader("Project Details")
            st.write("AI Advisor, an AI-driven financial advisor leveraging agent-based architecture and machine learning")
            st.write("**Type**: Hybrid Multi-Agent System")
            st.caption("All responses must be validated by experts.")
            st.caption("Artificial intelligence may not be accurate.")
            st.write("**Efficiency**: ≤2 LLM calls per analysis")
            
        else:
            st.error("❌ System initialization failed")
            st.write("Please check your .env file contains OPENAI_API_KEY")
    
    # Main content area
    if workspace_mode == "Problemas cotidianos":
        render_problem_solver_page()
        return

    if not st.session_state.analyzer:
        st.error("System not initialized. Please check your API key configuration.")
        return
    
    # File upload section
    st.header("📄 Upload Bank Statements")
    
    uploaded_files = st.file_uploader(
        "Choose one or more bank statement PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or many statements/cartolas for consolidated analysis"
    )
    
    if uploaded_files:
        st.info(f"📦 **Archivos seleccionados**: {len(uploaded_files)}")
        for up in uploaded_files:
            st.write(f"• {up.name} ({up.size:,} bytes)")

        statement_inputs = _build_statement_inputs_from_uploads(uploaded_files)

        # Analysis options
        col1, col2 = st.columns(2)
        
        with col1:
            generate_insights = st.checkbox(
                "🤖 Generate AI Insights",
                value=False,
                help="Use additional LLM call for personalized financial recommendations"
            )
        
        with col2:
            if generate_insights:
                st.info("📊 Se usarán llamadas LLM adicionales para recomendaciones")
            else:
                st.info("📊 Modo determinístico + categorización mínima con LLM")
        
        # Analysis button
        if st.button("🚀 Analyze Statements", type="primary", use_container_width=True):
            process_uploaded_files(statement_inputs, generate_insights)

        if st.session_state.analysis_result:
            show_debug_diagnostics(st.session_state.analysis_result)
            display_results(st.session_state.analysis_result)
    
    else:
        # Instructions when no file uploaded
        st.info("👆 Please upload a bank statement PDF to begin analysis")
        
        # Sample files section
        st.subheader("📋 Sample Files Available")
        st.write("You can test with these sample bank statements:")
        
        sample_files = [
            "chase_statement.pdf - Chase Bank format",
            "Sample Bank Statement.pdf - Generic format", 
            "Wells Fargo Statement.pdf - Wells Fargo format"
        ]
        
        for sample in sample_files:
            st.write(f"• {sample}")

def process_uploaded_files(statement_inputs: list, generate_insights: bool):
    """Process one or multiple uploaded files and show consolidated results."""
    temp_paths = []

    try:
        statements_for_analysis = []
        for idx, statement in enumerate(statement_inputs, start=1):
            uploaded_file = statement['uploaded_file']
            safe_name = Path(uploaded_file.name).name or f"statement_{idx}.pdf"
            suffix = Path(safe_name).suffix or ".pdf"
            prefix = re.sub(r'[^A-Za-z0-9_-]+', '_', Path(safe_name).stem)[:30] or f"statement_{idx}"
            with tempfile.NamedTemporaryFile(prefix=f"{prefix}_{idx}_", suffix=suffix, delete=False) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name
            temp_paths.append(temp_path)

            statements_for_analysis.append({
                'pdf_path': temp_path,
                'metadata': statement.get('metadata', {}),
            })

        progress_container = st.container()

        with progress_container:
            st.subheader("🔄 Processing Pipeline")
            progress_bar = st.progress(0)
            status_text = st.empty()

            agent_cols = st.columns(3)
            with agent_cols[0]:
                agent1_status = st.empty()
                agent1_status.info("🏗️ Agent 1: Waiting...")

            with agent_cols[1]:
                agent2_status = st.empty()
                agent2_status.info("🧠 Agent 2: Waiting...")

            with agent_cols[2]:
                agent3_status = st.empty()
                agent3_status.info("📊 Agent 3: Waiting...")

            status_text.text("🏗️ Agent 1: Processing PDFs...")
            agent1_status.warning("🏗️ Agent 1: Processing documents...")
            progress_bar.progress(20)

            status_text.text("🧠 Agent 2: Smart categorization...")
            agent1_status.success("🏗️ Agent 1: ✅ Complete")
            agent2_status.warning("🧠 Agent 2: Categorizing...")
            progress_bar.progress(60)

            if generate_insights:
                status_text.text("📊 Agent 3: Generating AI insights...")
            else:
                status_text.text("📊 Agent 3: Generating deterministic analysis...")

            agent2_status.success("🧠 Agent 2: ✅ Complete")
            agent3_status.warning("📊 Agent 3: Analyzing...")
            progress_bar.progress(90)

            result = st.session_state.analyzer.analyze_statements(
                statements_for_analysis,
                generate_ai_insights=generate_insights
            )

            agent3_status.success("📊 Agent 3: ✅ Complete")
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")

        if result['success']:
            st.session_state.analysis_result = result
            st.session_state.generate_ai_insights = generate_insights
            _clear_meta_analysis_state()
            st.success("🎉 Consolidated analysis completed successfully!")
        else:
            st.error(f"❌ Analysis failed: {result['error']}")
            st.info("Tip: Try cleaner PDF exports (text-based, not scanned image).")
            show_debug_diagnostics(result)

    except Exception as e:
        st.error(f"❌ Error processing files: {str(e)}")

    finally:
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)

def display_results(result: dict):
    """Display beautiful analysis results"""
    
    # Extract data
    analysis = result['analysis']
    summary = analysis['financial_summary']
    categories = analysis['category_breakdown']
    metrics = result['system_metrics']
    
    st.header("📊 Financial Analysis Results")

    all_transactions = result.get('transactions', [])
    available_persons = sorted({t.get('person', '') for t in all_transactions if t.get('person')})
    available_months = sorted({t.get('month', '') for t in all_transactions if t.get('month')})
    available_docs = sorted({t.get('source_file_name', '') for t in all_transactions if t.get('source_file_name')})
    parsed_dates = [
        parsed_date
        for parsed_date in (_parse_transaction_date(t.get('date')) for t in all_transactions)
        if parsed_date is not None
    ]
    default_start_date = min(parsed_dates) if parsed_dates else date.today()
    default_end_date = max(parsed_dates) if parsed_dates else date.today()

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        person_filter = st.selectbox("Persona", ["Todos"] + available_persons, key="results_filter_person")
    with filter_col2:
        document_filter = st.selectbox("Documento", ["Todos"] + available_docs, key="results_filter_document")
    with filter_col3:
        period_mode = st.selectbox(
            "Periodo",
            ["Todos", "Mes", "Rango de fechas"],
            key="results_filter_period_mode",
        )

    month_filter = "Todos"
    date_range = None
    if period_mode == "Mes":
        month_filter = st.selectbox("Mes", ["Todos"] + available_months, key="results_filter_month")
    elif period_mode == "Rango de fechas":
        selected_range = st.date_input(
            "Rango de fechas",
            value=(default_start_date, default_end_date),
            min_value=default_start_date,
            max_value=default_end_date,
            key="results_filter_date_range",
        )
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start_date, end_date = selected_range
            if start_date > end_date:
                st.warning("La fecha inicial no puede ser posterior a la fecha final.")
            else:
                date_range = (start_date, end_date)

    filtered_transactions = _filter_transactions(
        all_transactions,
        person_filter,
        month_filter,
        document_filter,
        date_filter_mode=period_mode,
        date_range=date_range,
    )
    if not filtered_transactions and all_transactions:
        st.warning("No hay transacciones para la combinación de filtros seleccionada.")

    visible_summary = _summarize_transactions(filtered_transactions)
    
    # Key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Total Spent",
            f"${visible_summary['total_spent']:.2f}",
            delta=f"{visible_summary['debit_count']} transactions"
        )
    
    with col2:
        st.metric(
            "💵 Total Income", 
            f"${visible_summary['total_income']:.2f}",
            delta=f"{visible_summary['credit_count']} deposits"
        )
    
    with col3:
        net_change = visible_summary['net_change']
        st.metric(
            "📊 Net Change",
            f"${net_change:.2f}",
            delta="Positive" if net_change > 0 else "Negative",
            delta_color="normal" if net_change > 0 else "inverse"
        )
    
    with col4:
        st.metric(
            "🤖 System Efficiency",
            f"{metrics['total_llm_calls']} LLM calls",
            delta=f"${metrics['estimated_cost']:.4f} cost"
        )

    st.caption(
        f"Transacciones visibles: {len(filtered_transactions)} / {len(all_transactions)} "
        f"(filtros aplicados por persona/documento/periodo)"
    )

    analyzer = st.session_state.get('analyzer')
    monthly_summary = analyzer.aggregate_by_month(filtered_transactions) if analyzer and filtered_transactions else []
    monthly_trends = analyzer.compute_monthly_trends(monthly_summary) if analyzer and monthly_summary else {}
    if monthly_summary:
        st.subheader("🗓️ Monthly Consolidation (visible period)")
        month_df = pd.DataFrame(monthly_summary)
        if not month_df.empty:
            st.dataframe(
                month_df[['month', 'total_income', 'total_spent', 'net_change', 'transaction_count']],
                use_container_width=True,
                hide_index=True,
            )

        if monthly_trends:
            trend_label = monthly_trends.get('classification', 'unknown')
            trend_details = monthly_trends.get('details', '')
            st.info(f"Tendencia: **{trend_label}**. {trend_details}")
    
    # Charts section
    if categories:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("🏷️ Spending by Category")
            
            # Build the pie from raw debit transactions so each slice reflects total spend.
            spending_totals = {}
            for txn in filtered_transactions:
                if txn.get('is_debit') and txn.get('effective_is_spending', True):
                    category_label = txn['category'].replace('_', ' ').title()
                    spending_totals[category_label] = spending_totals.get(category_label, 0.0) + float(txn['amount'])

            if not spending_totals:
                st.info("No hay gastos efectivos para graficar bajo los filtros actuales.")
            else:
                spending_rows = (
                    pd.DataFrame(
                        [{
                            'Category': category,
                            'Amount': float(amount)
                        } for category, amount in spending_totals.items()]
                    )
                    .sort_values('Amount', ascending=False)
                    .reset_index(drop=True)
                )

                category_names = spending_rows['Category'].tolist()
                category_amounts = spending_rows['Amount'].tolist()
                pie_colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Bold
                
                fig_pie = go.Figure(
                    data=[
                        go.Pie(
                            labels=category_names,
                            values=category_amounts,
                            sort=False,
                            direction='clockwise',
                            textposition='inside',
                            textinfo='percent+label',
                            marker=dict(
                                colors=pie_colors[:len(category_names)],
                                line=dict(color='white', width=2)
                            )
                        )
                    ]
                )
                fig_pie.update_layout(
                    title="Spending Distribution",
                    showlegend=True,
                    margin=dict(t=40, l=10, r=10, b=10)
                )
                if PLOTLY_EVENTS_AVAILABLE:
                    selected_points = plotly_events(
                        fig_pie,
                        click_event=True,
                        hover_event=False,
                        select_event=False,
                        key='spending_category_pie_events'
                    )

                    if selected_points:
                        point_index = selected_points[0].get('pointNumber')
                        if point_index is not None and 0 <= point_index < len(category_names):
                            _show_category_items_modal(category_names[point_index], filtered_transactions)
                else:
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.caption("Install 'streamlit-plotly-events' to enable click-to-open category modal.")
        
        with chart_col2:
            st.subheader("📈 Category Breakdown")
            
            filtered_debits = [
                t for t in filtered_transactions
                if t.get('is_debit') and t.get('effective_is_spending', True)
            ]

            if not filtered_debits:
                st.info("No hay gastos efectivos para mostrar desglose con los filtros actuales.")
            else:
                totals = {}
                counts = {}
                for txn in filtered_debits:
                    cat = txn.get('category', 'other')
                    totals[cat] = totals.get(cat, 0.0) + float(txn.get('amount') or 0.0)
                    counts[cat] = counts.get(cat, 0) + 1

                total_spent_filtered = sum(totals.values())
                rows = []
                for cat, amount in totals.items():
                    pct = (amount / total_spent_filtered * 100) if total_spent_filtered > 0 else 0.0
                    rows.append({
                        'Category': cat.replace('_', ' ').title(),
                        'Amount': amount,
                        'Percentage': pct,
                        'Count': counts.get(cat, 0),
                    })

                df_categories = pd.DataFrame(rows).sort_values('Amount', ascending=False).head(6)
                fig_bar = px.bar(
                    df_categories,
                    x='Category',
                    y='Amount',
                    title="Top Categories by Amount",
                    text='Amount',
                    color='Percentage',
                    color_continuous_scale='Viridis'
                )
                fig_bar.update_traces(texttemplate='$%{text:.0f}', textposition='outside')
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

    # Insights sections
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.subheader("💡 Financial Insights")
        for insight in analysis['basic_insights']:
            st.write(f"• {insight}")
    
    with insight_col2:
        if analysis.get('ai_insights') and analysis['ai_insights']:
            st.subheader("🤖 AI Recommendations")
            for insight in analysis['ai_insights'][:5]:  # Limit to 5
                if insight.strip():  # Only show non-empty insights
                    st.write(f"• {insight}")
        else:
            st.info("🤖 Enable 'Generate AI Insights' for personalized recommendations")
    
    # Transaction details (expandable)
    with st.expander("📋 View All Transactions", expanded=False):
        if all_transactions:
            st.caption("You can edit categories for any row. Then click 'Update report' to recalculate metrics, charts, and LLM insights.")

            st.caption("Puedes editar categorías por cartola: usa el filtro Documento para revisar cada archivo por separado. Los filtros de periodo solo cambian lo que ves.")

            if filtered_transactions:
                df_transactions = _transactions_to_editor_df(filtered_transactions)
                category_labels = [_category_code_to_label(code) for code in CATEGORY_CODES if code != 'uncategorized']

                edited_df = st.data_editor(
                    df_transactions,
                    use_container_width=True,
                    hide_index=True,
                    key='transactions_editor',
                    column_config={
                        '_txn_index': None,
                        'Amount': st.column_config.NumberColumn('Amount', format='$%.2f', disabled=True),
                        'Date': st.column_config.TextColumn('Date', disabled=True),
                        'Month': st.column_config.TextColumn('Month', disabled=True),
                        'Person': st.column_config.TextColumn('Person', disabled=True),
                        'Document': st.column_config.TextColumn('Document', disabled=True),
                        'Doc Type': st.column_config.TextColumn('Doc Type', disabled=True),
                        'Description': st.column_config.TextColumn('Description', disabled=True),
                        'Type': st.column_config.TextColumn('Type', disabled=True),
                        'Counts as spending': st.column_config.CheckboxColumn('Counts as spending', disabled=True),
                        'Internal transfer?': st.column_config.CheckboxColumn('Internal transfer?', disabled=True),
                        'Confidence': st.column_config.TextColumn('Confidence', disabled=True),
                        'Source': st.column_config.TextColumn('Source', disabled=True),
                        'Category': st.column_config.SelectboxColumn('Category', options=category_labels, required=True)
                    }
                )

                if st.button("🔄 Update report with manual categories", type="secondary", use_container_width=True):
                    _apply_manual_category_updates(result, edited_df)
            else:
                st.info("No hay transacciones que coincidan con los filtros actuales.")

    st.divider()
    st.subheader("🧠 Meta Analysis")
    st.caption("Run this only after reviewing and correcting transaction categories. This will make one additional LLM call using the knowledge base stored in data/advisory_knowledge_base_v1.json.")

    meta_button_disabled = st.session_state.get('analysis_result') is None
    if st.button("🚀 Run Meta Analysis", type="primary", use_container_width=True, disabled=meta_button_disabled):
        try:
            with st.spinner("Running meta analysis with knowledge base..."):
                meta_state = _run_meta_analysis(result)
                st.session_state.meta_analysis_result = meta_state['result']
                st.session_state.meta_analysis_metrics = meta_state['metrics']
                st.session_state.meta_analysis_source = meta_state['source_signature']
                st.session_state.meta_analysis_error = None
            st.success("Meta analysis completed successfully.")
        except Exception as e:
            st.session_state.meta_analysis_error = str(e)
            st.error(f"Meta analysis failed: {e}")

    meta_result = st.session_state.get('meta_analysis_result')
    meta_source = st.session_state.get('meta_analysis_source')
    current_source = analysis.get('report_generated_at')

    if meta_result and meta_source == current_source:
        _render_meta_analysis_result({
            'result': meta_result,
            'metrics': st.session_state.get('meta_analysis_metrics', {}),
        })
    elif st.session_state.get('meta_analysis_error'):
        st.warning(f"Last meta analysis attempt failed: {st.session_state.meta_analysis_error}")


def _apply_manual_category_updates(result: dict, edited_df: pd.DataFrame):
    """Apply manual category edits, persist learned rules, and recompute full analysis."""
    updated_transactions = copy.deepcopy(result['transactions'])
    changed_rows = 0
    rules_saved = 0

    for _, row in edited_df.iterrows():
        txn_index = int(row['_txn_index'])
        new_category = _category_label_to_code(row['Category'])
        old_category = updated_transactions[txn_index]['category']

        updated_transactions[txn_index]['category'] = new_category

        if new_category != old_category:
            changed_rows += 1
            saved_ok = st.session_state.analyzer.agent1.merchant_db.save_user_category_rule(
                updated_transactions[txn_index]['description'],
                new_category
            )
            if saved_ok:
                rules_saved += 1

    llm_before_calls = st.session_state.analyzer.llm.call_count
    llm_before_cost = st.session_state.analyzer.llm.total_cost

    # Reset Agent 3 counter only for this refresh run.
    st.session_state.analyzer.agent3.llm_calls_made = 0
    updated_analysis = st.session_state.analyzer.agent3.process(
        updated_transactions,
        generate_ai_insights=st.session_state.generate_ai_insights
    )

    llm_added_calls = st.session_state.analyzer.llm.call_count - llm_before_calls
    llm_added_cost = st.session_state.analyzer.llm.total_cost - llm_before_cost

    updated_result = copy.deepcopy(result)
    updated_result['transactions'] = updated_transactions
    updated_result['analysis'] = updated_analysis
    if hasattr(st.session_state.analyzer, 'aggregate_by_month'):
        updated_result['monthly_summary'] = st.session_state.analyzer.aggregate_by_month(updated_transactions)
    if hasattr(st.session_state.analyzer, 'compute_monthly_trends'):
        updated_result['monthly_trends'] = st.session_state.analyzer.compute_monthly_trends(updated_result.get('monthly_summary', []))
    updated_result['system_metrics']['total_llm_calls'] += max(llm_added_calls, 0)
    updated_result['system_metrics']['estimated_cost'] += max(llm_added_cost, 0.0)
    updated_result['system_metrics']['processing_time'] = datetime.now().isoformat()
    updated_result['system_metrics']['agent_breakdown']['agent3_llm_calls'] += max(llm_added_calls, 0)

    st.session_state.analysis_result = updated_result

    st.success(
        f"Updated report with {changed_rows} manual category change(s). "
        f"Saved {rules_saved} learned rule(s) for future statements."
    )
    if llm_added_calls > 0:
        st.info(f"LLM refresh used {llm_added_calls} additional call(s), estimated +${llm_added_cost:.4f}.")

    st.rerun()


def show_debug_diagnostics(result: dict):
    """Show parser diagnostics to help troubleshoot unsupported PDF formats."""
    debug_data = result.get('document_debug') or result.get('debug_info') or {}

    if isinstance(debug_data, list):
        with st.expander("🛠️ Parser Diagnostics", expanded=False):
            for item in debug_data:
                st.write(f"**{item.get('file_name', item.get('document_id', 'document'))}**")
                parsing_stats = item.get('parsing_stats')
                lines = item.get('sample_transaction_lines', [])[:10]
                if parsing_stats:
                    st.json(parsing_stats)
                if lines:
                    for line in lines:
                        st.code(_redact_diagnostic_line(line))
        return

    parsing_stats = debug_data.get('parsing_stats') or debug_data.get('document_processing', {}).get('parsing_stats')
    lines = debug_data.get('sample_transaction_lines') or debug_data.get('document_processing', {}).get('raw_transaction_lines', [])[:10]

    if not parsing_stats and not lines:
        return

    with st.expander("🛠️ Parser Diagnostics", expanded=False):
        if parsing_stats:
            st.write("**Parsing stats**")
            st.json(parsing_stats)

        if lines:
            st.write("**Detected transaction-like lines (sample)**")
            for line in lines:
                st.code(_redact_diagnostic_line(line))

if __name__ == "__main__":
    main()