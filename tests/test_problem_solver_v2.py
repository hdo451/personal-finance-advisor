"""
Pruebas unitarias para el módulo de solucionador de problemas financieros v2.
Valida: parser LLM, cálculos determinísticos, y lógica de UI.
"""

import pytest
import json
from utils.llm_problem_parser import LLMProblemParser
from utils.financial_calculator_v2 import (
    FinancialCalculator,
    AmortizationSystem,
    DepreciationMethod
)


class TestLLMProblemParser:
    """Pruebas del parser LLM."""
    
    def test_parser_initialization(self):
        """Verifica que el parser se inicializa correctamente."""
        parser = LLMProblemParser()
        assert parser.client is not None
    
    def test_parse_simple_problem(self):
        """Prueba parsing de un problema simple."""
        parser = LLMProblemParser()
        problem = "Tengo $10,000 a invertir a una tasa del 5% anual por 2 años"
        
        result = parser.parse_problem(problem)
        
        assert result.get("success") is True
        assert result.get("options") is not None
        assert len(result.get("options", [])) > 0
        assert result.get("problem_type") is not None
    
    def test_parse_problem_with_options(self):
        """Prueba parsing de problema con múltiples opciones."""
        parser = LLMProblemParser()
        problem = "Opción A: crédito de $50,000 a 8% por 3 años vs Opción B: leasing de $1,500 mensuales"
        
        result = parser.parse_problem(problem)
        
        assert result.get("success") is True
        # Debe detectar al menos 2 opciones
        assert len(result.get("options", [])) >= 2
    
    def test_parse_result_structure(self):
        """Verifica que la estructura del resultado es correcta."""
        parser = LLMProblemParser()
        problem = "$100,000 a 6% anual por 5 años"
        
        result = parser.parse_problem(problem)
        
        if result.get("success"):
            assert "options" in result
            assert "problem_type" in result
            assert "extraction_confidence" in result
            
            # Verificar estructura de opciones
            for option in result.get("options", []):
                assert "name" in option
                assert "parameters" in option
    
    def test_assumptions_marked_correctly(self):
        """Verifica que los supuestos se marcan correctamente."""
        parser = LLMProblemParser()
        problem = "$50,000 a 5% anual"  # Faltan períodos
        
        result = parser.parse_problem(problem)
        
        if result.get("success"):
            for option in result.get("options", []):
                for param_name, param_def in option.get("parameters", {}).items():
                    # Verificar que 'assumed' existe
                    assert "assumed" in param_def


class TestFinancialCalculator:
    """Pruebas del motor de cálculo."""
    
    def test_simple_interest(self):
        """Prueba cálculo de interés simple."""
        result = FinancialCalculator.calculate_simple_interest(
            principal=1000,
            rate=0.05,
            periods=2
        )
        
        assert result["pv"] == 1000
        assert result["fv"] == 1100  # 1000 + 1000*0.05*2
        assert result["interest"] == 100
    
    def test_compound_interest(self):
        """Prueba cálculo de interés compuesto."""
        result = FinancialCalculator.calculate_compound_interest(
            principal=1000,
            rate=0.05,
            periods=2
        )
        
        assert result["pv"] == 1000
        # FV = 1000 * (1.05)^2 = 1102.5
        assert abs(result["fv"] - 1102.5) < 0.01
    
    def test_payment_calculation(self):
        """Prueba cálculo de cuota (PMT)."""
        result = FinancialCalculator.calculate_payment(
            rate=0.05 / 12,  # Tasa mensual
            periods=60,      # 5 años
            pv=50000,        # Principal
            fv=0
        )
        
        assert result["pv"] == 50000
        assert result["payment"] > 0
        # Para $50,000 al 5% anual por 5 años, la cuota debe ser ~$943
        assert abs(result["payment"] - 943) < 50
    
    def test_amortization_schedule_french(self):
        """Prueba tabla de amortización sistema francés."""
        schedule = FinancialCalculator.calculate_amortization_schedule(
            principal=10000,
            rate=0.05,
            periods=5,
            system=AmortizationSystem.FRENCH
        )
        
        assert len(schedule) == 5
        
        # Verificar estructura
        for entry in schedule:
            assert "period" in entry
            assert "payment" in entry
            assert "interest" in entry
            assert "amortization" in entry
            assert "balance" in entry
        
        # El saldo final debe ser 0 (o muy cercano por redondeos)
        assert schedule[-1]["balance"] < 0.01
    
    def test_depreciation_straight_line(self):
        """Prueba depreciación línea recta."""
        schedule = FinancialCalculator.calculate_depreciation_schedule(
            cost=10000,
            residual_value=1000,
            useful_life=10,
            method=DepreciationMethod.STRAIGHT_LINE
        )
        
        assert len(schedule) == 10
        
        # Depreciación anual debe ser (10000-1000)/10 = 900
        for entry in schedule:
            assert abs(entry["depreciation"] - 900) < 0.01
        
        # Depreciación acumulada al final debe ser 9000
        assert abs(schedule[-1]["accumulated_depreciation"] - 9000) < 0.01
    
    def test_interest_rate_conversion(self):
        """Prueba conversión de tasas."""
        annual_rate = 0.05
        
        # Convertir de anual a mensual
        monthly = FinancialCalculator.convert_interest_rate(
            rate=annual_rate,
            from_period="annual",
            to_period="monthly"
        )
        
        assert monthly > 0
        # La tasa mensual debe ser menor que la anual
        assert monthly < annual_rate
    
    def test_irr_calculation(self):
        """Prueba cálculo de TIR (IRR)."""
        # Flujos: -100 (inversión inicial), 50, 50, 50
        cashflows = [-100, 50, 50, 50]
        
        irr = FinancialCalculator.calculate_irr(cashflows)
        
        assert irr > 0
        # Para estos flujos, IRR debe estar alrededor del 23%
        assert 0.1 < irr < 0.4
    
    def test_npv_calculation(self):
        """Prueba cálculo de VAN (NPV)."""
        rate = 0.10
        cashflows = [-100, 50, 50, 50]
        
        npv = FinancialCalculator.calculate_npv(rate, cashflows)
        
        # VAN positivo indica que la inversión es rentable
        assert isinstance(npv, float)


class TestIntegration:
    """Pruebas de integración."""
    
    def test_full_workflow(self):
        """Prueba el flujo completo: parsing -> validación -> cálculo."""
        parser = LLMProblemParser()
        
        # Paso 1: Parser
        problem = "$10,000 a 5% anual por 2 años"
        parsed = parser.parse_problem(problem)
        
        assert parsed.get("success") is True
        
        # Paso 2: Extraer parámetros (simulando validación)
        if parsed.get("options"):
            option = parsed["options"][0]
            params = option.get("parameters", {})
            
            # Paso 3: Calcular
            if params:
                principal = params.get("principal", {}).get("value", 0)
                rate = params.get("rate", {}).get("value", 0)
                periods = params.get("periods", {}).get("value", 1)
                
                if principal > 0 and rate > 0:
                    # Convertir a decimal si es necesario
                    if rate > 1:
                        rate = rate / 100
                    
                    result = FinancialCalculator.calculate_compound_interest(
                        principal=principal,
                        rate=rate,
                        periods=periods
                    )
                    
                    assert result["fv"] > principal


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
