"""
UI Streamlit para el solucionador de problemas financieros v2.
Implementa las 4 etapas: ingreso, extracción/validación, selección de columnas, resultados.
Módulo completamente aislado e independiente del resto del proyecto.
"""

import streamlit as st
import pandas as pd
from typing import Any
import json
import math
import html
import re
import unicodedata
import numpy_financial as npf
from utils.llm_problem_parser import LLMProblemParser
from utils.financial_calculator_v2 import (
    FinancialCalculator,
    AmortizationSystem,
    DepreciationMethod
)


DEFAULTS = {
    "principal": 10000.0,
    "pv": 10000.0,
    "rate": 0.10,
    "periods": 12.0,
    "fv": 0.0,
    "payment": 0.0,
    "residual_value": 0.0,
    "useful_life": 12.0,
}


def initialize_session_state():
    """Inicializa el estado de sesión."""
    if "stage" not in st.session_state:
        st.session_state.stage = 1
    if "problem_statement" not in st.session_state:
        st.session_state.problem_statement = ""
    if "parsed_problem" not in st.session_state:
        st.session_state.parsed_problem = None
    if "user_approved" not in st.session_state:
        st.session_state.user_approved = False
    if "selected_columns" not in st.session_state:
        st.session_state.selected_columns = []
    if "results" not in st.session_state:
        st.session_state.results = None
    if "problem_focus" not in st.session_state:
        st.session_state.problem_focus = "auto"


def stage_1_input():
    """Etapa 1: Ingreso del problema en lenguaje natural."""
    st.header("📝 Etapa 1: Ingresa tu problema")
    st.markdown("""
    Describe tu problema financiero en lenguaje natural. Puede incluir una o más opciones/alternativas.
    
    **Ejemplos:**
    - "Necesito un crédito de $100,000 a una tasa del 5% anual por 5 años vs arrendar por $2,000 mensuales"
    - "Compré una máquina por $50,000 con vida útil de 10 años y valor residual $5,000"
    """)
    
    problem = st.text_area(
        "Escribe tu problema aquí:",
        value=st.session_state.problem_statement,
        height=150,
        placeholder="Ej: Tengo dos opciones: crédito por $50k a 8% anual por 3 años vs leasing de $1500 mensuales..."
    )

    with st.expander("💡 Ayuda para encauzar el problema", expanded=False):
        st.markdown("Selecciona el tipo de tarea para guiar la extracción del LLM. Puedes dejarlo en automático si no estás seguro.")
        focus_options = {
            "auto": "Automático (que el sistema decida)",
            "comparar_creditos": "Comparar créditos / refinanciamiento",
            "prestamo": "Pedir un préstamo",
            "valor_presente": "Traer a valor presente",
            "valor_futuro": "Calcular valor futuro",
            "amortizacion": "Amortización / cuotas",
            "depreciacion": "Depreciación de activos",
            "flujos_caja": "Flujos de caja / VAN / TIR"
        }
        selected_focus = st.selectbox(
            "¿Qué estás tratando de resolver?",
            options=list(focus_options.keys()),
            format_func=lambda key: focus_options[key],
            index=list(focus_options.keys()).index(st.session_state.problem_focus)
            if st.session_state.problem_focus in focus_options else 0,
            key="problem_focus_selector"
        )
        st.session_state.problem_focus = selected_focus
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Analizar problema"):
            if not problem.strip():
                st.error("❌ Por favor, ingresa un problema")
                return
            
            st.session_state.problem_statement = problem
            
            # Parsear con LLM
            with st.spinner("🤔 Analizando con IA..."):
                try:
                    parser = LLMProblemParser()
                    parsed = parser.parse_problem(problem, problem_focus=st.session_state.problem_focus)
                    st.session_state.parsed_problem = parsed
                    
                    if parsed.get("success"):
                        st.session_state.stage = 2
                        st.rerun()
                    else:
                        st.error(f"❌ Error al parsear: {parsed.get('error')}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col2:
        if st.button("🔄 Limpiar"):
            st.session_state.problem_statement = ""
            st.session_state.parsed_problem = None
            st.session_state.stage = 1
            st.rerun()


def stage_2_validation():
    """Etapa 2: Extracción y validación humana con edición de valores."""
    st.header("✅ Etapa 2: Validar y editar parámetros")
    
    parsed = st.session_state.parsed_problem
    if not parsed or not parsed.get("success"):
        st.error("❌ Error en el parsing")
        return
    
    st.info(f"**Tipo de problema detectado:** {parsed.get('problem_type')}")
    st.info(f"**Confianza:** {parsed.get('extraction_confidence', 0):.0%}")

    def _is_numeric_like(value: Any) -> bool:
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            try:
                float(value)
                return True
            except ValueError:
                return False
        return False
    
    # Mostrar y permitir editar parámetros
    edited_options = []
    
    for i, option in enumerate(parsed.get("options", [])):
        with st.expander(f"📋 {option.get('name', f'Opción {i+1}')}", expanded=True):
            option_data = {"name": option.get("name"), "parameters": {}}
            
            params = option.get("parameters", {})
            for param_name, param_def in params.items():
                value = param_def.get("value")
                assumed = param_def.get("assumed", False)
                unit = param_def.get("unit", "")
                
                # Visual badge para valores supuestos
                label_text = f"{param_name} ({unit})"
                if assumed:
                    label_text += " 🔶 SUPUESTO"
                else:
                    label_text += " ✅ Explícito"
                
                # Input para editar el valor: numérico o textual según el dato detectado
                if _is_numeric_like(value):
                    numeric_value = float(value)
                    new_value = st.number_input(
                        label=label_text,
                        value=numeric_value,
                        key=f"param_{i}_{param_name}",
                        help=f"Valor original: {value} ({unit})"
                    )
                else:
                    new_value = st.text_input(
                        label=label_text,
                        value=str(value) if value is not None else "",
                        key=f"param_{i}_{param_name}",
                        help=f"Valor original: {value} ({unit})"
                    )
                
                option_data["parameters"][param_name] = {
                    "value": new_value,
                    "unit": unit,
                    "assumed": assumed
                }
            
            edited_options.append(option_data)
    
    # Guardae las ediciones en sesión
    if not hasattr(st.session_state, "validated_options"):
        st.session_state.validated_options = edited_options
    else:
        st.session_state.validated_options = edited_options
    
    # Mostrar supuestos realizados
    if parsed.get("assumptions_made"):
        with st.expander("📌 Supuestos realizados"):
            for assumption in parsed.get("assumptions_made", []):
                st.write(f"• {assumption}")
    
    # Botones de navegación
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("◀️ Volver"):
            st.session_state.stage = 1
            st.rerun()
    
    with col2:
        if st.button("✅ Aprobar y continuar"):
            st.session_state.user_approved = True
            st.session_state.stage = 3
            st.rerun()
    
    with col3:
        if st.button("🔄 Nueva consulta"):
            st.session_state.stage = 1
            st.session_state.parsed_problem = None
            st.session_state.problem_statement = ""
            st.rerun()


def stage_3_column_selection():
    """Etapa 3: Selección de columnas a mostrar en resultados."""
    st.header("🎯 Etapa 3: Selecciona columnas de salida")
    st.markdown("""
    Elige qué variables financieras deseas ver en la tabla de resultados.
    Solo se calcularán y mostrarán las columnas seleccionadas.
    """)

    st.markdown(
        """
        <style>
        .stCheckbox {
            margin-bottom: -0.35rem;
        }
        .stCheckbox label {
            padding-top: 0.1rem;
            padding-bottom: 0.1rem;
        }
        section[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] {
            gap: 0.15rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Organizar columnas por categorías
    categories = {
        "Valor del Dinero": [
            ("pv", "Valor Presente"),
            ("fv", "Valor Futuro"),
        ],
        "Tasas de Interés": [
            ("rate", "Tasa de Interés"),
            ("interest_rate_nominal", "Tasa Nominal"),
            ("interest_rate_effective", "Tasa Efectiva"),
        ],
        "Pagos y Cuotas": [
            ("payment", "Cuota"),
            ("total_paid", "Total Pagado"),
        ],
        "Amortización": [
            ("amortization", "Amortización"),
            ("interest_paid", "Intereses Pagados"),
            ("balance", "Saldo"),
        ],
        "Depreciación": [
            ("residual_value", "Valor Residual"),
            ("accumulated_depreciation", "Depreciación Acumulada"),
        ],
        "Períodos": [
            ("periods", "Número de Períodos"),
        ],
    }
    
    selected = []
    for category, columns in categories.items():
        st.markdown(f"**📊 {category}**")
        for col_key, col_label in columns:
            if st.checkbox(col_label, key=f"col_{col_key}"):
                selected.append(col_key)
    
    st.session_state.selected_columns = selected
    
    # Botones
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("◀️ Volver"):
            st.session_state.stage = 2
            st.rerun()
    
    with col2:
        if st.button("▶️ Generar resultados"):
            if not selected:
                st.error("❌ Selecciona al menos una columna")
                return
            
            st.session_state.stage = 4
            st.rerun()
    
    with col3:
        if st.button("🔄 Nueva consulta"):
            st.session_state.stage = 1
            st.session_state.parsed_problem = None
            st.session_state.problem_statement = ""
            st.session_state.selected_columns = []
            st.rerun()


def _get_param_value(params: dict[str, Any], key: str, aliases: list[str]) -> tuple[float | None, str, str | None]:
    """Obtiene un parámetro y retorna valor, estado y razón.

    Estados:
    - explicit: provisto por usuario/enunciado
    - llm_assumed: marcado por parser como supuesto
    - missing: no disponible
    """
    candidates = [key] + aliases

    # 1) Preferir valores explícitos aunque existan claves genéricas asumidas.
    for candidate in candidates:
        if candidate in params:
            param_def = params.get(candidate, {})
            raw_value = param_def.get("value")
            if raw_value is None:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            assumed = bool(param_def.get("assumed", False))
            if not assumed:
                return value, "explicit", None

    # 2) Si no hay explícitos, usar el primero asumido encontrado.
    for candidate in candidates:
        if candidate in params:
            param_def = params.get(candidate, {})
            raw_value = param_def.get("value")
            if raw_value is None:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if bool(param_def.get("assumed", False)):
                reason = f"{candidate} provino de supuesto en etapa 2"
                return value, "llm_assumed", reason
            return value, "explicit", None

    return None, "missing", None


def _get_param_unit(params: dict[str, Any], key: str, aliases: list[str]) -> str:
    """Obtiene unidad textual del parámetro si existe."""
    for candidate in [key] + aliases:
        if candidate in params:
            unit = params.get(candidate, {}).get("unit", "")
            return str(unit).lower().strip()
    return ""


def _is_assumed_status(status: str) -> bool:
    return status in {"llm_assumed", "default_assumed"}


def _join_assumption_reasons(*reason_items: tuple[str, str | None]) -> str:
    reasons: list[str] = []
    for status, reason in reason_items:
        if _is_assumed_status(status) and reason:
            reasons.append(reason)
    unique_reasons = list(dict.fromkeys(reasons))
    if unique_reasons:
        return "Supuestos usados: " + "; ".join(unique_reasons)
    return "Calculado con datos explícitos y/o inferencias determinísticas"


def _normalize_rate_to_period(rate_value: float, rate_unit: str, periods_unit: str, periods_value: float | None) -> float:
    """Convierte tasa a la periodicidad de cálculo (mensual/anual) cuando aplica."""
    rate = rate_value
    if rate > 1:
        rate = rate / 100.0

    ru = (rate_unit or "").lower()
    pu = (periods_unit or "").lower()

    # Si no hay unidad explícita, heurística mínima:
    # tasas altas con períodos largos suelen ser anuales.
    if not ru and periods_value is not None and periods_value >= 12 and rate >= 0.20:
        ru = "annual"

    monthly_markers = ["month", "months", "mes", "meses", "monthly"]
    annual_markers = ["year", "years", "anio", "año", "anual", "annual", "yearly"]

    periods_monthly = any(marker in pu for marker in monthly_markers)
    periods_annual = any(marker in pu for marker in annual_markers)

    rate_annual = any(marker in ru for marker in annual_markers)
    rate_monthly = any(marker in ru for marker in monthly_markers)

    if rate_annual and periods_monthly:
        return FinancialCalculator.convert_interest_rate(rate, "annual", "monthly")
    if rate_monthly and periods_annual:
        return FinancialCalculator.convert_interest_rate(rate, "monthly", "annual")
    return rate


def _format_cell_value(row_key: str, value: float) -> str:
    """Formatea el valor para visualización final."""
    if row_key in {"rate", "interest_rate_nominal", "interest_rate_effective"}:
        return f"{value * 100:,.2f}%"
    return f"{value:,.2f}"


def _infer_global_asset_price(problem_statement: str) -> float | None:
    """Infiere un valor de activo global desde el enunciado (usa el monto monetario más alto razonable)."""
    if not problem_statement:
        return None

    text = problem_statement.lower()
    money_matches = re.findall(r"\$?\s*([0-9][0-9\.,]{3,})", text)
    candidates: list[float] = []
    for raw in money_matches:
        token = raw.strip()

        # Normalizar miles/decimales con heurística robusta.
        if "," in token and "." in token:
            if token.rfind(".") > token.rfind(","):
                # 1,000,000.00 -> quitar comas
                normalized = token.replace(",", "")
            else:
                # 1.000.000,00 -> quitar puntos y usar punto decimal
                normalized = token.replace(".", "").replace(",", ".")
        elif "," in token:
            comma_parts = token.split(",")
            if len(comma_parts) > 2:
                # 1,480,000 -> miles
                normalized = token.replace(",", "")
            else:
                # 1234,56 o 123,4
                if len(comma_parts[-1]) <= 2:
                    normalized = token.replace(",", ".")
                else:
                    normalized = token.replace(",", "")
        elif "." in token:
            dot_parts = token.split(".")
            if len(dot_parts) > 2:
                # 1.480.000 -> miles
                normalized = token.replace(".", "")
            else:
                normalized = token
        else:
            normalized = token

        try:
            value = float(normalized)
        except ValueError:
            continue
        if 50000 <= value <= 20000000:
            candidates.append(value)

    if not candidates:
        return None

    # Heurística: en estos casos el costo del activo suele ser el mayor monto puntual.
    return max(candidates)


def _normalize_ascii(text: str) -> str:
    """Quita acentos para comparar textos de forma robusta."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()


def _build_option_outputs(
    option: dict[str, Any],
    global_asset_price: float | None = None,
    problem_focus: str | None = None,
) -> tuple[dict[str, float], dict[str, bool], dict[str, str]]:
    """Calcula salidas de una opción, priorizando inferencias antes de usar supuestos por defecto."""
    params = option.get("parameters", {})

    principal, principal_status, principal_reason = _get_param_value(params, "principal", ["capital", "monto", "purchase_price", "credit_amount"])
    pv, pv_status, pv_reason = _get_param_value(params, "pv", ["principal", "capital", "monto", "purchase_price", "credit_amount"])
    rate, rate_status, rate_reason = _get_param_value(params, "rate", ["interest_rate", "tasa", "annual_rate", "monthly_rate"])
    periods, periods_status, periods_reason = _get_param_value(params, "periods", ["n", "term", "plazo", "remaining_payments", "remaining_periods", "months_left"])
    fv, fv_status, fv_reason = _get_param_value(params, "fv", ["future_value", "valor_futuro", "valor_final"])
    payment, payment_status, payment_reason = _get_param_value(params, "payment", ["pmt", "cuota", "monthly_payment", "lease_payment", "rent_payment"])
    total_repayment, total_repayment_status, total_repayment_reason = _get_param_value(
        params,
        "total_repayment",
        ["total_payment", "total_paid", "monto_total", "monto_total_devolver", "valor_final", "payoff_amount", "payoff", "total_obligation", "amount_due", "total_payoff", "total_payment_obligation"]
    )
    purchase_price, purchase_price_status, purchase_price_reason = _get_param_value(
        params,
        "purchase_price",
        ["asset_price", "precio_compra", "machine_price", "precio_maquina", "pv", "principal"]
    )
    credit_amount, credit_amount_status, credit_amount_reason = _get_param_value(
        params,
        "credit_amount",
        ["loan_amount", "monto_credito", "principal", "pv"]
    )
    maintenance_cost, maintenance_cost_status, maintenance_cost_reason = _get_param_value(
        params,
        "maintenance_cost",
        ["maintenance", "monthly_maintenance", "mantencion", "mantenimiento"]
    )
    residual_value, residual_status, residual_reason = _get_param_value(params, "residual_value", ["salvage_value", "valor_residual"])

    rate_unit = _get_param_unit(params, "rate", ["interest_rate", "tasa", "annual_rate", "monthly_rate"])
    periods_unit = _get_param_unit(params, "periods", ["n", "term", "plazo"])

    option_name = _normalize_ascii(str(option.get("name", "")))
    is_leasing = "leasing" in option_name
    is_rent = "arrend" in option_name or "renta" in option_name
    is_purchase = "compra" in option_name
    focus_key = _normalize_ascii(problem_focus or "")
    comparison_focus = focus_key == "comparar_creditos"

    # rate_output: lo que se muestra en tabla (respeta unidad ingresada)
    # rate_calc: tasa por período utilizada en cálculos determinísticos
    rate_output: float | None = None
    rate_calc: float | None = None

    # Normalización básica
    if pv is None and principal is not None and principal > 0:
        pv = principal
        pv_status = principal_status
        pv_reason = principal_reason

    if rate is not None:
        rate_output = rate / 100.0 if rate > 1 else rate
        rate_calc = _normalize_rate_to_period(rate, rate_unit, periods_unit, periods)

    if rate_calc is not None and rate_calc <= 0:
        rate_calc = None
    if rate_output is not None and rate_output <= 0:
        rate_output = None
    if periods is not None and periods <= 0:
        periods = None
        periods_status = "missing"
        periods_reason = None
    if pv is not None and pv <= 0:
        pv = None
        pv_status = "missing"
        pv_reason = None
    if fv is not None and fv <= 0:
        fv = None
        fv_status = "missing"
        fv_reason = None
    if payment is not None and payment <= 0:
        payment = None
        payment_status = "missing"
        payment_reason = None
    if maintenance_cost is not None and maintenance_cost < 0:
        maintenance_cost = None
        maintenance_cost_status = "missing"
        maintenance_cost_reason = None
    if total_repayment is not None and total_repayment <= 0:
        total_repayment = None
        total_repayment_status = "missing"
        total_repayment_reason = None

    # Si viene total_repayment explícito y FV no viene, se usa como FV final.
    if fv is None and total_repayment is not None:
        fv = total_repayment
        fv_status = total_repayment_status
        fv_reason = total_repayment_reason or "FV tomado desde total_repayment"

    # Si hay precio de compra explícito, se usa como PV/base cuando faltan variables principales.
    if purchase_price is not None and (pv is None or pv <= 0):
        pv = purchase_price
        pv_status = purchase_price_status
        pv_reason = purchase_price_reason or "PV tomado desde purchase_price"

    # Fallback: usar costo global del activo inferido del enunciado.
    if (pv is None or pv <= 0) and global_asset_price is not None and global_asset_price > 0:
        pv = global_asset_price
        pv_status = "calculated"
        pv_reason = "PV inferido desde el costo global del activo en el enunciado"

    # Si hay crédito explícito, úsalo como base financiada si no hay principal/pv claro.
    financed_principal = credit_amount if credit_amount is not None and credit_amount > 0 else pv
    if financed_principal is not None and (principal is None or principal <= 0):
        principal = financed_principal
        principal_status = credit_amount_status if credit_amount is not None else pv_status
        principal_reason = credit_amount_reason if credit_amount is not None else pv_reason

    # 1) Inferencias determinísticas antes de suponer
    if rate_calc is None and pv is not None and fv is not None and periods is not None and periods > 0 and pv > 0 and fv > 0:
        inferred_rate = (fv / pv) ** (1.0 / periods) - 1.0
        rate_calc = inferred_rate
        rate_output = inferred_rate
        rate_status = "calculated"
        rate_reason = "Tasa calculada desde PV, FV y períodos"

    if periods is None and pv is not None and fv is not None and rate_calc is not None and rate_calc > 0 and pv > 0 and fv > 0:
        periods = math.log(fv / pv) / math.log(1.0 + rate_calc)
        periods_status = "calculated"
        periods_reason = "Períodos calculados desde PV, FV y tasa"

    if pv is None and fv is not None and rate_calc is not None and periods is not None and rate_calc > -1 and periods > 0:
        pv = fv / ((1.0 + rate_calc) ** periods)
        pv_status = "calculated"
        pv_reason = "PV calculado desde FV, tasa y períodos"

    if fv is None and pv is not None and rate_calc is not None and periods is not None and periods > 0:
        fv = pv * ((1.0 + rate_calc) ** periods)
        fv_status = "calculated"
        fv_reason = "FV calculado desde PV, tasa y períodos"

    is_loan_like = is_purchase or ("credito" in option_name) or ("préstamo" in option_name) or ("prestamo" in option_name)

    if rate_calc is None and is_loan_like and pv is not None and payment is not None and periods is not None and periods > 0:
        try:
            solved_rate = float(npf.rate(int(round(periods)), -payment, pv, 0))
            if not math.isnan(solved_rate) and solved_rate > -1:
                rate_calc = solved_rate
                rate_output = solved_rate
                rate_status = "calculated"
                rate_reason = "Tasa calculada desde PV, cuota y períodos"
        except Exception:
            pass

    if payment is None and financed_principal is not None and rate_calc is not None and periods is not None and periods > 0:
        payment_calc = FinancialCalculator.calculate_payment(
            rate=rate_calc,
            periods=periods,
            pv=financed_principal,
            fv=0
        )
        payment = float(payment_calc.get("payment", 0.0))
        payment_status = "calculated"
        payment_reason = "Cuota calculada desde PV, tasa y períodos"

    # 2) Fallback final con supuestos por defecto (solo si no se pudo inferir y realmente hace falta calcular algo)
    if pv is None:
        pv = DEFAULTS["pv"]
        pv_status = "default_assumed"
        pv_reason = f"PV asumido por defecto = {DEFAULTS['pv']}"
    if rate_calc is None and total_repayment is None and payment is None:
        rate_calc = DEFAULTS["rate"]
        rate_output = DEFAULTS["rate"]
        rate_status = "default_assumed"
        rate_reason = f"Tasa asumida por defecto = {DEFAULTS['rate'] * 100:.2f}%"
    if periods is None:
        periods = DEFAULTS["periods"]
        periods_status = "default_assumed"
        periods_reason = f"Períodos asumidos por defecto = {DEFAULTS['periods']:.0f}"
    if fv is None and rate_calc is not None:
        fv = pv * ((1.0 + rate_calc) ** periods)
        fv_status = "calculated"
        fv_reason = "FV calculado desde PV, tasa y períodos"
    if payment is None and rate_calc is not None:
        payment_calc = FinancialCalculator.calculate_payment(
            rate=rate_calc,
            periods=periods,
            pv=financed_principal if financed_principal is not None else pv,
            fv=0
        )
        payment = float(payment_calc.get("payment", 0.0))
        payment_status = "calculated"
        payment_reason = "Cuota calculada desde PV, tasa y períodos"

    # Mantenimiento mensual se suma a la cuota si corresponde (leasing/renta).
    payment_effective = payment
    if maintenance_cost is not None and maintenance_cost > 0 and (is_leasing or is_rent):
        payment_effective = payment + maintenance_cost
        # Solo marcar supuesto si mantenimiento venía supuesto
        if _is_assumed_status(maintenance_cost_status):
            payment_status = "default_assumed"
            payment_reason = maintenance_cost_reason or "Incluye mantención asumida"
    if residual_value is None or residual_value <= 0:
        residual_value = pv * 0.10
        residual_status = "default_assumed"
        residual_reason = "Valor residual asumido por defecto = 10% de PV"

    # Métricas agregadas para tabla
    upfront = 0.0
    upfront_status = "calculated"
    upfront_reason = "Sin pago inicial adicional"
    if purchase_price is not None and financed_principal is not None and purchase_price > financed_principal:
        upfront = purchase_price - financed_principal
        upfront_status = "explicit" if (not _is_assumed_status(purchase_price_status) and not _is_assumed_status(credit_amount_status)) else "llm_assumed"
        upfront_reason = "Pago inicial = precio de compra - monto financiado"

    if total_repayment is not None:
        total_paid = total_repayment
        total_paid_status = total_repayment_status
        total_paid_reason = total_repayment_reason or "Total pagado tomado desde total_repayment"
    else:
        if is_leasing:
            # Leasing: cuota + mantención durante plazo + residual final si aplica.
            residual_component = residual_value if residual_value is not None and residual_value > 0 else 0.0
            total_paid = (payment_effective * periods) + residual_component
            total_paid_reason = "Total pagado leasing = (cuota + mantención) x períodos + residual"
        elif is_rent:
            # Renta operativa: cuota (+ mantención si está explícita) x períodos.
            total_paid = payment_effective * periods
            total_paid_reason = "Total pagado renta = (cuota + mantención) x períodos"
        elif is_purchase:
            # Compra con mezcla de reservas + crédito.
            total_paid = upfront + (payment_effective * periods)
            total_paid_reason = "Total pagado compra = pago inicial + (cuota crédito x períodos)"
        else:
            total_paid = payment_effective * periods if payment_effective > 0 else fv
            total_paid_reason = "Total pagado calculado desde cuota x períodos o FV"
        total_paid_status = "calculated"

    # En modo comparar créditos, si existen cuota mensual + pagos restantes,
    # el valor futuro/costo restante debe ser la suma de cuotas pendientes.
    has_schedule = (payment_effective is not None and payment_effective > 0 and periods is not None and periods > 0)
    if comparison_focus and has_schedule and ("monthly_payment" in params or "remaining_payments" in params):
        scheduled_total = payment_effective * periods
        fv = scheduled_total
        fv_status = "calculated"
        fv_reason = "FV calculado como cuota mensual x cuotas pendientes en modo comparar créditos"
        total_paid = scheduled_total
        total_paid_status = "calculated"
        total_paid_reason = "Total pagado calculado como cuota mensual x cuotas pendientes en modo comparar créditos"

    # Si el usuario pide FV en una estructura de financiamiento y no existe un FV explícito,
    # mostrar el total pagado calculado para que la tabla refleje el costo futuro real.
    if fv is None and total_paid is not None and (is_loan_like or is_leasing or is_rent or is_purchase):
        fv = total_paid
        fv_status = total_paid_status
        fv_reason = total_paid_reason or "FV mostrado como total pagado calculado"

    interest_paid = max(total_paid - pv, 0.0)
    amortization = pv / periods if periods > 0 else 0.0
    balance = max(pv - amortization * periods, 0.0)
    accumulated_depreciation = max(pv - residual_value, 0.0)
    nominal_rate = rate_calc * 12 if rate_calc is not None else None
    effective_rate = FinancialCalculator.convert_interest_rate(rate_calc, "monthly", "annual") if rate_calc is not None else None

    if rate_output is None:
        rate_output = rate_calc

    if rate_unit and periods_unit and rate_output is not None and rate_calc is not None and abs(rate_output - rate_calc) > 1e-9:
        rate_reason = (
            f"Tasa ingresada {rate_output * 100:.2f}% ({rate_unit}) "
            f"convertida a {rate_calc * 100:.2f}% por período ({periods_unit}) para cálculo"
        )

    outputs = {
        "pv": pv,
        "fv": fv,
        "rate": rate_output,
        "periods": periods,
        "payment": payment_effective,
        "balance": balance,
        "amortization": amortization,
        "interest_paid": interest_paid,
        "residual_value": residual_value,
        "accumulated_depreciation": accumulated_depreciation,
        "interest_rate_nominal": nominal_rate,
        "interest_rate_effective": effective_rate,
        "total_paid": total_paid,
    }

    input_status = {
        "pv": (pv_status, pv_reason),
        "rate": (rate_status, rate_reason),
        "periods": (periods_status, periods_reason),
        "fv": (fv_status, fv_reason),
        "payment": (payment_status, payment_reason),
        "total_paid": (total_paid_status, total_paid_reason),
        "total_repayment": (total_repayment_status, total_repayment_reason),
        "residual_value": (residual_status, residual_reason),
    }

    dependencies = {
        "pv": ["pv"],
        "fv": ["fv", "pv", "rate", "periods"],
        "rate": ["rate", "pv", "fv", "periods", "payment"],
        "periods": ["periods", "pv", "fv", "rate"],
        "payment": ["payment", "pv", "rate", "periods"],
        "balance": ["pv", "periods"],
        "amortization": ["pv", "periods"],
        "interest_paid": ["payment", "pv", "rate", "periods"],
        "residual_value": ["residual_value", "pv"],
        "accumulated_depreciation": ["residual_value", "pv"],
        "interest_rate_nominal": ["rate"],
        "interest_rate_effective": ["rate"],
        "total_paid": ["total_paid", "total_repayment", "payment", "periods", "fv"],
    }

    output_assumed: dict[str, bool] = {}
    output_tooltips: dict[str, str] = {}

    # Reglas específicas para que outputs directos (ej. total_repayment) no hereden supuestos irrelevantes.
    output_status_reason = {
        "pv": (pv_status, pv_reason),
        "fv": (fv_status, fv_reason),
        "rate": (rate_status, rate_reason),
        "periods": (periods_status, periods_reason),
        "payment": (payment_status, payment_reason),
        "residual_value": (residual_status, residual_reason),
        "total_paid": (total_paid_status, total_paid_reason),
    }

    for out_key, deps in dependencies.items():
        if out_key in output_status_reason:
            status, reason = output_status_reason[out_key]
            output_assumed[out_key] = _is_assumed_status(status)
            output_tooltips[out_key] = _join_assumption_reasons((status, reason))
            continue

        dep_items = [input_status.get(dep, ("missing", None)) for dep in deps]
        assumed = any(_is_assumed_status(status) for status, _ in dep_items)
        output_assumed[out_key] = assumed
        output_tooltips[out_key] = _join_assumption_reasons(*dep_items)

    return outputs, output_assumed, output_tooltips


def stage_4_results():
    """Etapa 4: Tabla de resultados comparativos."""
    st.header("📈 Etapa 4: Resultados")
    
    parsed = st.session_state.parsed_problem
    validated_options = st.session_state.validated_options
    selected_columns = st.session_state.selected_columns
    problem_statement = st.session_state.get("problem_statement", "")
    
    if not selected_columns:
        st.error("❌ No hay columnas seleccionadas")
        return
    
    column_names = FinancialCalculator.AVAILABLE_COLUMNS
    row_keys = [col for col in selected_columns if col in column_names]
    row_labels = [column_names.get(col, col) for col in row_keys]

    if not validated_options:
        st.warning("⚠️ No hay opciones para calcular")
        return

    # Construcción orientada al requerimiento: columnas=opciones, filas=variables
    results_matrix: dict[str, dict[str, float]] = {label: {} for label in row_labels}
    assumption_matrix: dict[str, dict[str, bool]] = {label: {} for label in row_labels}
    tooltip_matrix: dict[str, dict[str, str]] = {label: {} for label in row_labels}

    global_asset_price = _infer_global_asset_price(problem_statement)

    for option in validated_options:
        option_name = option.get("name") or "Opción"
        try:
            outputs, output_assumed, output_tooltips = _build_option_outputs(
                option,
                global_asset_price=global_asset_price,
                problem_focus=st.session_state.get("problem_focus", None),
            )
            for key, label in zip(row_keys, row_labels):
                value = outputs.get(key)
                if value is None:
                    assumed = False
                    tip = "Supuesto usado: valor faltante, se aplicó fallback de seguridad"
                else:
                    assumed = bool(output_assumed.get(key, True))
                    tip = output_tooltips.get(key, "")

                results_matrix[label][option_name] = value
                assumption_matrix[label][option_name] = assumed
                tooltip_matrix[label][option_name] = tip
        except Exception as e:
            st.warning(f"⚠️ Error al calcular {option_name}: {str(e)}")
            for key, label in zip(row_keys, row_labels):
                results_matrix[label][option_name] = None
                assumption_matrix[label][option_name] = False
                tooltip_matrix[label][option_name] = "Supuesto usado: error de cálculo, se aplicó fallback"

    if results_matrix:
        df = pd.DataFrame.from_dict(results_matrix, orient="index")
        df = df.apply(pd.to_numeric, errors="coerce")

        assumed_df = pd.DataFrame.from_dict(assumption_matrix, orient="index").reindex_like(df).fillna(False)
        tooltip_df = pd.DataFrame.from_dict(tooltip_matrix, orient="index").reindex_like(df).fillna("")

        st.caption("🔶 Las celdas amarillas indican que el resultado usó uno o más datos supuestos.")

        # Render HTML custom para evitar artefactos visuales de Styler (span class="pd-t") y mantener tooltip por celda.
        label_to_key = {label: key for key, label in zip(row_keys, row_labels)}
        html_rows: list[str] = []
        for row_label in df.index:
            row_key = label_to_key.get(row_label, "")
            cells = [f"<th style='text-align:left; padding:10px; border:1px solid #e5e7eb; background:#f9fafb;'>{html.escape(str(row_label))}</th>"]
            for col_name in df.columns:
                raw_value = df.loc[row_label, col_name]
                if pd.isna(raw_value):
                    text_value = "N/A"
                else:
                    value = float(raw_value)
                    text_value = _format_cell_value(row_key, value)
                tooltip = str(tooltip_df.loc[row_label, col_name])
                assumed = bool(assumed_df.loc[row_label, col_name])
                bg = "#fff3cd" if assumed else "#ffffff"
                color = "#7a4f01" if assumed else "#111827"
                cells.append(
                    "<td "
                    f"title='{html.escape(tooltip)}' "
                    "style='padding:10px; border:1px solid #e5e7eb; text-align:right; "
                    f"background:{bg}; color:{color}; font-weight:600;'>{html.escape(text_value)}</td>"
                )
            html_rows.append("<tr>" + "".join(cells) + "</tr>")

        headers = [f"<th style='text-align:left; padding:10px; border:1px solid #e5e7eb; background:#f3f4f6;'></th>"]
        for col_name in df.columns:
            headers.append(
                f"<th style='text-align:left; padding:10px; border:1px solid #e5e7eb; background:#f3f4f6;'>{html.escape(str(col_name))}</th>"
            )

        table_html = (
            "<div style='overflow-x:auto;'>"
            "<table style='border-collapse:collapse; width:100%; font-size:0.98rem;'>"
            f"<thead><tr>{''.join(headers)}</tr></thead>"
            f"<tbody>{''.join(html_rows)}</tbody>"
            "</table>"
            "</div>"
        )
        st.markdown(table_html, unsafe_allow_html=True)
        
        # Descargar resultados
        csv = df.to_csv()
        st.download_button(
            label="📥 Descargar como CSV",
            data=csv,
            file_name="resultados_financieros.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ No se generaron resultados")
    
    # Botones finales
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("◀️ Editar parámetros"):
            st.session_state.stage = 2
            st.rerun()
    
    with col2:
        if st.button("🔄 Nueva consulta"):
            st.session_state.stage = 1
            st.session_state.parsed_problem = None
            st.session_state.problem_statement = ""
            st.session_state.selected_columns = []
            st.session_state.user_approved = False
            st.rerun()
    
    with col3:
        if st.button("⚙️ Mostrar JSON"):
            st.json({
                "parsed_problem": parsed,
                "validated_options": validated_options,
                "selected_columns": selected_columns
            })


def render_problem_solver_page():
    """Renderiza el módulo independiente dentro de otra app Streamlit."""
    st.title("🧮 Solucionador de Problemas Financieros v2")
    st.markdown("""
    **Módulo independiente** para analizar y resolver problemas financieros complejos.
    
    El sistema usa IA para interpretar tu problema en lenguaje natural, y luego ejecuta
    cálculos determinísticos exactos para generar resultados reproducibles.
    """)
    
    initialize_session_state()
    
    # Mostrar indicador de etapa actual
    progress_bar = st.progress((st.session_state.stage - 1) / 4)
    st.markdown(f"**Etapa {st.session_state.stage} de 4**")
    
    # Ejecutar etapa actual
    if st.session_state.stage == 1:
        stage_1_input()
    elif st.session_state.stage == 2:
        stage_2_validation()
    elif st.session_state.stage == 3:
        stage_3_column_selection()
    elif st.session_state.stage == 4:
        stage_4_results()


def main():
    """Punto de entrada de la aplicación independiente."""
    st.set_page_config(
        page_title="🧮 Solucionador de Problemas Financieros v2",
        layout="wide"
    )
    render_problem_solver_page()


if __name__ == "__main__":
    main()
