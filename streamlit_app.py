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
from datetime import datetime
import os
import copy
from pathlib import Path
from dotenv import load_dotenv

try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_AVAILABLE = True
except ImportError:
    PLOTLY_EVENTS_AVAILABLE = False

# Import your system
from main_coordinator import BankStatementAnalyzer
from utils.financial_solver import solve_problem

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
            '_txn_index': idx,
            'Date': txn['date'],
            'Description': txn['description'],
            'Category': _category_code_to_label(txn['category']),
            'Amount': txn['amount'],
            'Type': 'OUT' if txn['is_debit'] else 'IN',
            'Confidence': f"{txn['confidence']:.0%}",
            'Source': txn['source'].title()
        })
    return pd.DataFrame(rows)

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

    system_prompt = """You are a financial analyst.
Write a concise but human-readable report in Spanish with this structure:
1) Interpretación del problema
2) Supuestos y defaults utilizados
3) Pasos de cálculo (claros)
4) Resultado principal
5) Interpretación ejecutiva
6) Recomendaciones accionables

Rules:
- Mention that this is educational and not regulated investment advice.
- Do not recommend specific financial products without user profile context.
- Always reference the deterministic results provided in the input.
"""

    user_prompt = f"""Original question:
{question}

Deterministic solution trace:
{json.dumps(solved, ensure_ascii=False, indent=2)}
"""

    narrative = llm.make_call(user_prompt, system_prompt, expect_json=False)
    if not narrative:
        raise ValueError("No se pudo generar redacción ejecutiva.")

    calls_used = llm.call_count - calls_before
    cost_used = llm.total_cost - cost_before
    return {
        "text": narrative,
        "metrics": {
            "llm_calls": max(calls_used, 0),
            "estimated_cost": max(cost_used, 0.0),
        },
    }


def render_problem_solver_page():
    st.header("🧮 Problemas cotidianos")
    st.caption("Resuelve preguntas financieras en lenguaje natural con cálculo determinístico en Python y redacción ejecutiva asistida por LLM.")

    question = st.text_area(
        "Describe tu problema financiero",
        placeholder="Ej: ¿Me conviene crédito A o B para comprar auto?"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        currency = st.selectbox("Moneda", ["MXN", "USD"], index=0)
    with col2:
        st.text_input("Periodicidad", value="monthly", disabled=True)
    with col3:
        st.text_input("Base de días", value="365", disabled=True)

    if st.button("🧠 Interpretar problema", type="secondary", use_container_width=True):
        if not question.strip():
            st.warning("Escribe una pregunta antes de interpretar.")
        else:
            try:
                with st.spinner("Interpretando y ordenando datos..."):
                    parsed = _interpret_problem_to_json(question.strip(), currency)
                    st.session_state.problem_source_question = question.strip()
                    st.session_state.problem_draft = parsed["draft"]
                    st.session_state.problem_draft_text = json.dumps(parsed["draft"], ensure_ascii=False, indent=2)
                    st.session_state.problem_solver_metrics = {
                        "parse_llm_calls": parsed["metrics"]["llm_calls"],
                        "parse_cost": parsed["metrics"]["estimated_cost"],
                        "narrative_llm_calls": 0,
                        "narrative_cost": 0.0,
                    }
                    st.session_state.problem_solver_error = None
            except Exception as e:
                st.session_state.problem_solver_error = str(e)

    if st.session_state.get("problem_draft_text"):
        st.subheader("📋 Datos reordenados (requiere autorización humana)")
        st.caption("Revisa y edita el JSON antes de resolver. El cálculo solo corre cuando autorizas.")

        draft_dict = st.session_state.get("problem_draft", {})
        defaults_used = draft_dict.get("defaults_used", [])
        
        if defaults_used:
            defaults_markdown = "".join([
                f"<li style='margin-bottom: 8px;'><span style='color:#b00020; font-weight:700;'>{d}</span></li>"
                for d in defaults_used
            ])
            st.markdown(
                f"""
                <div style="background-color: #fff0f0; border: 2px solid #d32f2f; color: #b00020; padding: 16px; border-radius: 8px; margin-bottom: 15px;">
                    <strong style="font-size: 1.05rem;">⚠️ ATENCIÓN - DATOS ASUMIDOS AUTOMÁTICAMENTE:</strong><br>
                    <span style="font-weight: 700;">El problema no contenía toda la información necesaria.</span>
                    Se inyectaron estos supuestos para poder calcular. Revisa cada uno y modifícalo si deseas trabajar con otro valor:<br><br>
                    <ul style="margin-top: 8px; margin-bottom: 0; padding-left: 22px;">
                        {defaults_markdown}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        edited = st.text_area(
            "JSON autorizado",
            value=st.session_state.problem_draft_text,
            height=360,
            key="problem_solver_json_editor",
        )

        if st.button("✅ Autorizar y resolver", type="primary", use_container_width=True):
            try:
                authorized_payload = json.loads(edited)
                solved = _solve_authorized_problem(authorized_payload)
                narrative = _generate_problem_narrative(question.strip(), solved)

                metrics = st.session_state.problem_solver_metrics or {
                    "parse_llm_calls": 0,
                    "parse_cost": 0.0,
                    "narrative_llm_calls": 0,
                    "narrative_cost": 0.0,
                }
                metrics["narrative_llm_calls"] = narrative["metrics"]["llm_calls"]
                metrics["narrative_cost"] = narrative["metrics"]["estimated_cost"]
                metrics["total_llm_calls"] = metrics["parse_llm_calls"] + metrics["narrative_llm_calls"]
                metrics["total_cost"] = metrics["parse_cost"] + metrics["narrative_cost"]

                st.session_state.problem_solver_result = solved
                st.session_state.problem_solver_narrative = narrative["text"]
                st.session_state.problem_solver_metrics = metrics
                st.session_state.problem_solver_error = None
            except Exception as e:
                st.session_state.problem_solver_error = str(e)

    if st.session_state.get("problem_solver_error"):
        st.error(f"Error en resolución: {st.session_state.problem_solver_error}")

    solved = st.session_state.get("problem_solver_result")
    if solved:
        metrics = st.session_state.get("problem_solver_metrics", {})

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("LLM interpretación", metrics.get("parse_llm_calls", 0), f"${metrics.get('parse_cost', 0.0):.4f}")
        with m2:
            st.metric("LLM redacción", metrics.get("narrative_llm_calls", 0), f"${metrics.get('narrative_cost', 0.0):.4f}")
        with m3:
            st.metric("Costo total etapa", f"{metrics.get('total_llm_calls', 0)} calls", f"${metrics.get('total_cost', 0.0):.4f}")

        st.subheader("🧾 Trazabilidad")
        st.json(
            {
                "inputs": solved.get("trace", {}).get("inputs", {}),
                "assumptions": solved.get("trace", {}).get("assumptions", []),
                "formula": solved.get("trace", {}).get("formula", ""),
                "result": solved.get("trace", {}).get("result", {}),
                "timestamp": solved.get("timestamp"),
            }
        )

        st.subheader("📄 Respuesta ejecutiva")
        st.markdown(st.session_state.get("problem_solver_narrative", ""))
        st.caption("Supuestos y defaults usados están documentados en la trazabilidad superior.")


def _generate_executive_summary(result: dict, income: float, expenses: float, savings: float, savings_rate: float, health_status: str) -> str:
    """Generate a professional narrative executive summary"""
    
    categories = result.get('category_analysis', [])
    ratios = result.get('ratios', {})
    
    # Calculate category totals
    fixed_categories = ['bills_utilities', 'fees']
    variable_categories = ['groceries', 'transportation']
    
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
    
    fixed_pct = (total_fixed / expenses * 100) if expenses > 0 else 0
    variable_pct = (total_variable / expenses * 100) if expenses > 0 else 0
    discretionary_pct = (total_discretionary / expenses * 100) if expenses > 0 else 0
    
    # Build narrative
    summary = []
    
    # Opening
    if health_status == "good":
        summary.append("**Situación Financiera General:** Tu perfil financiero muestra una posición sólida y estable. "
                      "Con ingresos mensuales de ${:.2f} y gastos de ${:.2f}, logras un ahorro neto de ${:.2f} "
                      "(equivalente al {:.1f}% de tus ingresos), lo que te posiciona en una trayectoria de crecimiento patrimonial."
                      .format(income, expenses, savings, savings_rate * 100))
    elif health_status == "risk":
        summary.append("**Situación Financiera General:** Tu perfil financiero requiere atención inmediata. "
                      "Actualmente gastas ${:.2f} de cada ${:.2f} que ganas, dejando tan solo ${:.2f} de ahorro mensual ({:.1f}%). "
                      "Existen oportunidades claras de optimización que deben abordarse con urgencia."
                      .format(expenses, income, savings, savings_rate * 100))
    else:
        summary.append("**Situación Financiera General:** Tu perfil financiero es neutral. "
                      "Con ingresos de ${:.2f} y gastos de ${:.2f}, logras un ahorro de ${:.2f} ({:.1f}%). "
                      "Hay margen para mejora en varios aspectos de tu gestión financiera."
                      .format(income, expenses, savings, savings_rate * 100))
    
    # Expense breakdown narrative
    summary.append("\n**Estructura de Gastos:** Tu gasto total se distribuye en tres categorías clave. "
                  "Los **gastos fijos** (vivienda, servicios, seguros) representan **{:.0f}% del presupuesto** (${:.2f}), "
                  "lo que está dentro de rangos saludables. Los **gastos variables** (alimentos, transporte) constituyen **{:.0f}%** (${:.2f}), "
                  "reflejando necesidades cotidianas regulares. Finalmente, los **gastos discrecionales** (entretenimiento, compras, suscripciones) "
                  "alcanzan **{:.0f}%** (${:.2f})."
                  .format(fixed_pct, total_fixed, variable_pct, total_variable, discretionary_pct, total_discretionary))
    
    # Key insights
    if discretionary_pct > 25:
        summary.append("\n**Hallazgo Principal:** Se identifica un nivel significativo de gasto discrecional ({:.0f}%), "
                      "lo que representa la principal área de optimización. Este segmento incluye suscripciones, compras impulsivas, "
                      "entretenimiento y dining out. Reducir este categoría solo al 15-20% del presupuesto podría liberar "
                      "${:.2f} adicionales mensuales para ahorros e inversión."
                      .format(discretionary_pct, total_discretionary * 0.40))
    elif discretionary_pct > 15:
        summary.append("\n**Hallazgo Principal:** El gasto discrecional está en un nivel moderado ({:.0f}%). "
                      "Aunque controlado, aún existe potencial para mejorar mediante pequeños ajustes en suscripciones "
                      "y entretenimiento, liberando aproximadamente ${:.2f} mensuales."
                      .format(discretionary_pct, total_discretionary * 0.25))
    else:
        summary.append("\n**Hallazgo Principal:** El gasto discrecional está bien controlado en {:.0f}%. "
                      "Mantienes una disciplina sólida en áreas de discreción, lo que facilita tu capacidad de ahorro."
                      .format(discretionary_pct))
    
    # Recommendations section
    summary.append("\n**Recomendaciones Estratégicas:** Basado en este análisis, se sugieren las siguientes acciones: "
                  "(1) Revisar la sección de **Recomendaciones** arriba para estrategias específicas; "
                  "(2) Implementar un **presupuesto categorizado** usando la proporción 50/30/20 adaptada a tu perfil; "
                  "(3) Automatizar transferencias de ahorro tan pronto se recibe el ingreso; "
                  "(4) Monitorear regularmente el **Análisis de Gastos por Categoría** para detectar desviaciones temprano.")
    
    # Closing
    summary.append("\n**Próximos Pasos:** Para traducir este análisis en mejoras concretas, consulta la sección de "
                  "**Ratios Financieros** para contexto de tu posición relativa, revisa los **Problemas Detectados** "
                  "si existen, y considera implementar las acciones listadas en **Recomendaciones**. Vuelve a generar este "
                  "reporte mensualmente para monitorear tu progreso hacia tus objetivos financieros.")
    
    return "\n".join(summary)


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
    st.markdown(executive_summary)
    
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
        api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key:
            with st.spinner("🏗️ Initializing 3-agent system..."):
                st.session_state.analyzer = BankStatementAnalyzer(api_key)
            st.success("✅ System ready!")
        else:
            st.session_state.analyzer = None
            st.error("❌ Please set OPENAI_API_KEY in .env file")

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
        st.header("🤖 System Status")
        
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
            st.write("Aiadvisor, AI-driven financial advisor leveraging agent-based architecture and machine learning")
            st.write("**Type**: Hybrid Multi-Agent System")
            st.caption("All responses must be validated by experts.")
            st.caption("Artifitial intellillence may not be accurate.")
            st.write("**Efficiency**: ≤2 LLM calls per analysis")
            
        else:
            st.error("❌ System initialization failed")
            st.write("Please check your .env file contains OPENAI_API_KEY")

        st.divider()
        workspace_mode = st.radio(
            "Módulo",
            ["Análisis de cartola", "Problemas cotidianos"],
            index=0,
            help="Selecciona entre análisis de estados de cuenta y resolución de problemas financieros cotidianos"
        )
    
    # Main content area
    if not st.session_state.analyzer:
        st.error("System not initialized. Please check your API key configuration.")
        return

    if workspace_mode == "Problemas cotidianos":
        render_problem_solver_page()
        return
    
    # File upload section
    st.header("📄 Upload Bank Statement")
    
    uploaded_file = st.file_uploader(
        "Choose a bank statement PDF file",
        type=['pdf'],
        help="Upload your bank statement PDF for automated analysis"
    )
    
    if uploaded_file:
        # Show file info
        st.info(f"📄 **File**: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
        
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
                st.info("📊 Will use 2 LLM calls (~$0.004)")
            else:
                st.info("📊 Will use 1 LLM call (~$0.002)")
        
        # Analysis button
        if st.button("🚀 Analyze Statement", type="primary", use_container_width=True):
            process_uploaded_file(uploaded_file, generate_insights)

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

def process_uploaded_file(uploaded_file, generate_insights: bool):
    """Process the uploaded file and show results"""
    
    # Save uploaded file temporarily
    temp_path = f"temp_{uploaded_file.name}"
    
    try:
        # Write uploaded file to disk
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Create processing progress
        progress_container = st.container()
        
        with progress_container:
            st.subheader("🔄 Processing Pipeline")
            
            # Progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Agent processing steps
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
            
            # Step 1: Agent 1
            status_text.text("🏗️ Agent 1: Processing PDF...")
            agent1_status.warning("🏗️ Agent 1: Processing PDF...")
            progress_bar.progress(20)
            
            # Step 2: Agent 2
            status_text.text("🧠 Agent 2: Smart categorization...")
            agent1_status.success("🏗️ Agent 1: ✅ Complete")
            agent2_status.warning("🧠 Agent 2: Categorizing...")
            progress_bar.progress(60)
            
            # Step 3: Agent 3
            if generate_insights:
                status_text.text("📊 Agent 3: Generating AI insights...")
            else:
                status_text.text("📊 Agent 3: Generating analysis...")
            
            agent2_status.success("🧠 Agent 2: ✅ Complete") 
            agent3_status.warning("📊 Agent 3: Analyzing...")
            progress_bar.progress(90)
            
            # Run the actual analysis
            result = st.session_state.analyzer.analyze_statement(
                temp_path,
                generate_ai_insights=generate_insights
            )
            
            # Complete
            agent3_status.success("📊 Agent 3: ✅ Complete")
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
        
        # Display results
        if result['success']:
            st.session_state.analysis_result = result
            st.session_state.generate_ai_insights = generate_insights
            _clear_meta_analysis_state()
            st.success("🎉 Analysis completed successfully!")
        else:
            st.error(f"❌ Analysis failed: {result['error']}")
            st.info("Tip: If this is a local bank statement/cartola, try a cleaner PDF export (text-based, not scanned image).")
            show_debug_diagnostics(result)
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
    
    finally:
        # Clean up temporary file
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
    
    # Key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Total Spent",
            f"${summary['total_spent']:.2f}",
            delta=f"{summary['debit_count']} transactions"
        )
    
    with col2:
        st.metric(
            "💵 Total Income", 
            f"${summary['total_income']:.2f}",
            delta=f"{summary['credit_count']} deposits"
        )
    
    with col3:
        net_change = summary['net_change']
        st.metric(
            "📊 Net Change",
            f"${net_change:.2f}",
            delta="Positive" if net_change > 0 else "Negative",
            delta_color="normal" if net_change > 0 else "inverse"
        )
    
    with col4:
        st.metric(
            "🤖 System Efficiency",
            f"{metrics['total_llm_calls']}/2 LLM calls",
            delta=f"${metrics['estimated_cost']:.4f} cost"
        )
    
    # Charts section
    if categories:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("🏷️ Spending by Category")
            
            # Build the pie from raw debit transactions so each slice reflects total spend.
            spending_totals = {}
            for txn in result['transactions']:
                if txn['is_debit']:
                    category_label = txn['category'].replace('_', ' ').title()
                    spending_totals[category_label] = spending_totals.get(category_label, 0.0) + float(txn['amount'])

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
                        _show_category_items_modal(category_names[point_index], result['transactions'])
            else:
                st.plotly_chart(fig_pie, use_container_width=True)
                st.caption("Install 'streamlit-plotly-events' to enable click-to-open category modal.")
        
        with chart_col2:
            st.subheader("📈 Category Breakdown")
            
            # Create bar chart
            df_categories = pd.DataFrame([
                {
                    'Category': cat['category'].replace('_', ' ').title(),
                    'Amount': cat['total'],
                    'Percentage': cat['percentage'],
                    'Count': cat['transaction_count']
                }
                for cat in categories[:6]  # Top 6 categories
            ])
            
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
        if result['transactions']:
            st.caption("You can edit categories for any row. Then click 'Update report' to recalculate metrics, charts, and LLM insights.")

            df_transactions = _transactions_to_editor_df(result['transactions'])
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
                    'Description': st.column_config.TextColumn('Description', disabled=True),
                    'Type': st.column_config.TextColumn('Type', disabled=True),
                    'Confidence': st.column_config.TextColumn('Confidence', disabled=True),
                    'Source': st.column_config.TextColumn('Source', disabled=True),
                    'Category': st.column_config.SelectboxColumn('Category', options=category_labels, required=True)
                }
            )

            if st.button("🔄 Update report with manual categories", type="secondary", use_container_width=True):
                _apply_manual_category_updates(result, edited_df)

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
                st.code(line)

if __name__ == "__main__":
    main()