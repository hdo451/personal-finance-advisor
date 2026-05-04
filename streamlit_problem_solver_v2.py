"""
UI Streamlit para el solucionador de problemas financieros v2.
Implementa las 4 etapas: ingreso, extracción/validación, selección de columnas, resultados.
Módulo completamente aislado e independiente del resto del proyecto.
"""

import streamlit as st
import pandas as pd
from typing import Any
import json
from utils.llm_problem_parser import LLMProblemParser
from utils.financial_calculator_v2 import (
    FinancialCalculator,
    AmortizationSystem,
    DepreciationMethod
)


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
                    parsed = parser.parse_problem(problem)
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
                
                # Input para editar el valor
                new_value = st.number_input(
                    label=label_text,
                    value=float(value) if value else 0.0,
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
        st.subheader(f"📊 {category}")
        cols = st.columns(2)
        for idx, (col_key, col_label) in enumerate(columns):
            with cols[idx % 2]:
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


def stage_4_results():
    """Etapa 4: Tabla de resultados comparativos."""
    st.header("📈 Etapa 4: Resultados")
    
    parsed = st.session_state.parsed_problem
    validated_options = st.session_state.validated_options
    selected_columns = st.session_state.selected_columns
    
    if not selected_columns:
        st.error("❌ No hay columnas seleccionadas")
        return
    
    # Construir tabla de resultados
    results_data = {}
    
    for option in validated_options:
        option_name = option.get("name", "Opción")
        results_data[option_name] = {}
        
        params = option.get("parameters", {})
        
        # Intentar ejecutar cálculos según el tipo de problema
        try:
            problem_type = parsed.get("problem_type", "")
            
            # Mapear valores de parámetros
            principal = params.get("principal", {}).get("value", 0)
            rate = params.get("rate", {}).get("value", 0)
            periods = params.get("periods", {}).get("value", 12)
            fv = params.get("fv", {}).get("value", 0)
            payment = params.get("payment", {}).get("value", 0)
            
            # Convertir tasa a decimal si es porcentaje
            if rate > 1:
                rate = rate / 100
            
            # Ejecutar cálculo según el tipo
            if problem_type in ["simple_interest", "compound_interest"]:
                calc = FinancialCalculator.calculate_compound_interest(
                    principal=principal,
                    rate=rate,
                    periods=periods
                )
            elif problem_type in ["annuity", "amortization"]:
                calc = FinancialCalculator.calculate_payment(
                    rate=rate,
                    periods=periods,
                    pv=principal,
                    fv=fv
                )
            else:
                calc = {}
            
            # Mapear resultados a columnas seleccionadas
            for col_key in selected_columns:
                if col_key in calc:
                    results_data[option_name][col_key] = calc.get(col_key, "N/A")
        
        except Exception as e:
            st.warning(f"⚠️ Error al calcular {option_name}: {str(e)}")
    
    # Convertir a DataFrame
    if results_data:
        df = pd.DataFrame(results_data).T
        
        # Renombrar columnas
        column_names = FinancialCalculator.AVAILABLE_COLUMNS
        df.columns = [column_names.get(col, col) for col in df.columns]
        
        # Mostrar tabla
        st.dataframe(df.round(4), use_container_width=True)
        
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


def main():
    """Punto de entrada de la aplicación."""
    st.set_page_config(
        page_title="🧮 Solucionador de Problemas Financieros v2",
        layout="wide"
    )
    
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


if __name__ == "__main__":
    main()
