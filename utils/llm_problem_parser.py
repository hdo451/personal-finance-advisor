"""
LLM Problem Parser — Extrae estructura de problemas financieros en lenguaje natural.

Responsabilidades:
- Detectar opciones/alternativas en el enunciado
- Extraer parámetros para cada opción
- Identificar valores supuestos vs explícitos
- Devolver JSON estructurado con toda la metadata
"""

import json
import os
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMProblemParser:
    """Parser que usa LLM para estructurar problemas financieros."""
    
    FINANCIAL_PROBLEM_TYPES = [
        "simple_interest",
        "compound_interest",
        "annuity",
        "amortization",
        "depreciation",
        "cashflow",
        "comparison"
    ]
    
    STANDARD_ASSUMPTIONS = {
        "periods": 12,  # meses
        "rate_type": "monthly",
        "depreciation_method": "straight_line",
        "amortization_system": "french"
    }
    
    def __init__(self):
        """Inicializa el cliente de OpenAI."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY no está configurada en .env")
        self.client = OpenAI(api_key=api_key)
    
    def parse_problem(self, problem_statement: str, problem_focus: str | None = None) -> dict[str, Any]:
        """
        Parsea un enunciado de problema financiero.
        
        Args:
            problem_statement: Enunciado en lenguaje natural del problema
            
        Returns:
            Dict con estructura: {
                "success": bool,
                "error": str (si falla),
                "problem_type": str,
                "options": [
                    {
                        "name": str,
                        "parameters": {
                            "param_name": {
                                "value": Any,
                                "unit": str,
                                "assumed": bool
                            }
                        }
                    }
                ],
                "raw_extraction": str (respuesta del LLM para debugging)
            }
        """
        try:
            # Construir el prompt para el LLM
            prompt = self._build_extraction_prompt(problem_statement, problem_focus=problem_focus)
            
            # Llamar al LLM
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """Eres un experto en análisis de problemas financieros.
Tu tarea es:
1. Identificar todas las opciones/alternativas en el problema
2. Extraer los parámetros para cada opción
3. Identificar qué valores fueron explícitos vs cuáles debes asumir
4. Devolver SOLO un JSON válido sin explicaciones adicionales

Formato de salida esperado:
{
    "problem_type": "annuity|amortization|depreciation|comparison|etc",
    "options": [
        {
            "name": "Opción A",
            "parameters": {
                "principal": {"value": 10000, "unit": "currency", "assumed": false},
                "rate": {"value": 0.05, "unit": "annual", "assumed": false},
                "periods": {"value": 12, "unit": "months", "assumed": true}
            }
        }
    ],
    "extraction_confidence": 0.95,
    "missing_critical_parameters": [],
    "assumptions_made": ["periods = 12 meses (estándar)"]
}"""
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            # Extraer JSON de la respuesta
            raw_text = response.choices[0].message.content
            parsed_json = self._extract_json_from_response(raw_text)
            
            if not parsed_json:
                return {
                    "success": False,
                    "error": "No se pudo extraer JSON válido de la respuesta del LLM",
                    "raw_extraction": raw_text
                }
            
            return {
                "success": True,
                "problem_type": parsed_json.get("problem_type"),
                "options": parsed_json.get("options", []),
                "extraction_confidence": parsed_json.get("extraction_confidence", 0),
                "missing_critical_parameters": parsed_json.get("missing_critical_parameters", []),
                "assumptions_made": parsed_json.get("assumptions_made", []),
                "raw_extraction": raw_text
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Error en el parser LLM: {str(e)}"
            }
    
    def _build_extraction_prompt(self, problem_statement: str, problem_focus: str | None = None) -> str:
        """Construye el prompt para el LLM."""
        focus_instructions = ""
        if problem_focus:
            focus_map = {
                "comparar_creditos": "Prioriza comparar costo total futuro, cuotas, plazo, tasa y cualquier pago inicial o residual entre opciones de crédito/leasing/refinanciamiento.",
                "prestamo": "Prioriza identificar monto financiado, tasa, plazo, cuota y monto total a pagar.",
                "valor_presente": "Prioriza identificar el flujo futuro, tasa y plazo para traer el valor a presente.",
                "valor_futuro": "Prioriza identificar el valor inicial, tasa y plazo para proyectar el valor futuro.",
                "amortizacion": "Prioriza cuota, tasa, plazo, saldo, capital e intereses para el sistema de amortización.",
                "depreciacion": "Prioriza costo del activo, vida útil, valor residual y método de depreciación.",
                "flujos_caja": "Prioriza identificar flujos de caja, fechas, tasas y cualquier VAN/TIR relevante."
            }
            focus_label = {
                "comparar_creditos": "comparar créditos / refinanciamiento",
                "prestamo": "pedir un préstamo",
                "valor_presente": "traer a valor presente",
                "valor_futuro": "calcular valor futuro",
                "amortizacion": "amortización",
                "depreciacion": "depreciación",
                "flujos_caja": "flujos de caja"
            }.get(problem_focus, problem_focus)
            focus_instructions = f"""
FOCO DE INTERPRETACIÓN SELECCIONADO POR EL USUARIO: {focus_label}
INSTRUCCIÓN ESPECIAL:
{focus_map.get(problem_focus, '')}

"""

        return f"""Analiza el siguiente problema financiero y extrae su estructura en JSON:

PROBLEMA:
{problem_statement}

{focus_instructions}

INSTRUCCIONES:
1. Identifica TODAS las opciones/alternativas mencionadas (ej: "opción A", "préstamo", "leasing", etc)
2. Para cada opción, extrae los parámetros mencionados (capital, tasa, períodos, etc)
3. Si un parámetro se omitió pero es necesario para el cálculo, usa estos valores estándar y marca como "assumed": true:
   - Períodos: 12 meses
   - Tipo de tasa: mensual
   - Método de depreciación: línea recta
   - Sistema de amortización: francés

4. IMPORTANTE: Marca cada parámetro con "assumed": false si fue explícito en el enunciado, true si fue asumido

5. Devuelve SOLO el JSON, sin explicaciones adicionales."""
    
    def _extract_json_from_response(self, response_text: str) -> dict[str, Any] | None:
        """Extrae JSON de la respuesta del LLM."""
        try:
            # Intentar parsear directamente
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Intentar extraer JSON de un bloque de código
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                if end > start:
                    json_str = response_text[start:end].strip()
                    return json.loads(json_str)
            
            # Intentar extraer el primer { ... }
            try:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    return json.loads(json_str)
            except:
                pass
        
        return None
    
    def validate_and_apply_defaults(self, parsed: dict[str, Any]) -> dict[str, Any]:
        """
        Valida la estructura parseada y aplica valores por defecto donde falten.
        
        Args:
            parsed: Salida de parse_problem()
            
        Returns:
            Dict validado con todos los defaults aplicados
        """
        if not parsed.get("success"):
            return parsed
        
        # Aplicar valores por defecto a parámetros faltantes
        for option in parsed.get("options", []):
            for param_name, param_def in option.get("parameters", {}).items():
                if "unit" not in param_def:
                    param_def["unit"] = "unknown"
                if "assumed" not in param_def:
                    param_def["assumed"] = False
        
        return parsed
