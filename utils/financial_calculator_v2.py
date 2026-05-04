"""
Motor de cálculo determinístico para problemas financieros.

Utiliza numpy-financial y scipy para cálculos exactos y reproducibles.
No hay intervención del LLM — solo matemática pura.
"""

import numpy as np
import numpy_financial as npf
from scipy import optimize
from typing import Any
from enum import Enum


class AmortizationSystem(Enum):
    """Sistemas de amortización soportados."""
    FRENCH = "french"           # Cuota fija
    GERMAN = "german"           # Amortización fija
    AMERICAN = "american"       # Interés + capital final


class DepreciationMethod(Enum):
    """Métodos de depreciación soportados."""
    STRAIGHT_LINE = "straight_line"
    DOUBLE_DECLINING = "double_declining"
    SUM_OF_DIGITS = "sum_of_digits"


class FinancialCalculator:
    """Motor de cálculo para problemas financieros."""
    
    # Salidas posibles en tablas de resultado
    AVAILABLE_COLUMNS = {
        "pv": "Valor Presente",
        "fv": "Valor Futuro",
        "rate": "Tasa de Interés",
        "periods": "Número de Períodos",
        "payment": "Cuota",
        "balance": "Saldo",
        "amortization": "Amortización",
        "interest_paid": "Intereses Pagados",
        "residual_value": "Valor Residual",
        "accumulated_depreciation": "Depreciación Acumulada",
        "interest_rate_nominal": "Tasa Nominal",
        "interest_rate_effective": "Tasa Efectiva",
        "total_paid": "Total Pagado"
    }
    
    @staticmethod
    def calculate_simple_interest(
        principal: float,
        rate: float,
        periods: float
    ) -> dict[str, float]:
        """
        Calcula interés simple.
        
        Args:
            principal: Capital inicial
            rate: Tasa de interés (decimal, ej 0.05 para 5%)
            periods: Número de períodos
            
        Returns:
            Dict con pv, fv, interest, total
        """
        interest = principal * rate * periods
        fv = principal + interest
        return {
            "pv": principal,
            "fv": fv,
            "interest": interest,
            "total": fv,
            "rate": rate,
            "periods": periods
        }
    
    @staticmethod
    def calculate_compound_interest(
        principal: float,
        rate: float,
        periods: float,
        compounding_periods: int = 1
    ) -> dict[str, float]:
        """
        Calcula interés compuesto.
        
        Args:
            principal: Capital inicial
            rate: Tasa de interés anual (decimal)
            periods: Número de períodos
            compounding_periods: Capitalizaciones por año
            
        Returns:
            Dict con pv, fv, interest
        """
        fv = npf.fv(
            rate=rate / compounding_periods,
            nper=periods,
            pmt=0,
            pv=-principal
        )
        interest = fv - principal
        return {
            "pv": principal,
            "fv": fv,
            "interest": interest,
            "rate": rate,
            "periods": periods,
            "compounding_periods": compounding_periods
        }
    
    @staticmethod
    def calculate_annuity(
        rate: float,
        periods: float,
        pmt: float,
        when: str = "end"
    ) -> dict[str, float]:
        """
        Calcula valor presente/futuro de una anualidad.
        
        Args:
            rate: Tasa de interés por período (decimal)
            periods: Número de períodos
            pmt: Pago periódico
            when: "end" para anualidad ordinaria, "begin" para anualidad adelantada
            
        Returns:
            Dict con pv, fv, payment
        """
        when_numeric = 0 if when == "end" else 1
        pv = npf.pv(
            rate=rate,
            nper=periods,
            pmt=-pmt,
            when=when_numeric
        )
        fv = npf.fv(
            rate=rate,
            nper=periods,
            pmt=-pmt,
            when=when_numeric
        )
        return {
            "pv": abs(pv),
            "fv": fv,
            "payment": pmt,
            "rate": rate,
            "periods": periods
        }
    
    @staticmethod
    def calculate_payment(
        rate: float,
        periods: float,
        pv: float,
        fv: float = 0,
        when: str = "end"
    ) -> dict[str, float]:
        """
        Calcula la cuota periódica (PMT).
        
        Args:
            rate: Tasa de interés por período (decimal)
            periods: Número de períodos
            pv: Valor presente (principal)
            fv: Valor futuro deseado
            when: "end" o "begin"
            
        Returns:
            Dict con payment y otros parámetros
        """
        when_numeric = 0 if when == "end" else 1
        pmt = npf.pmt(
            rate=rate,
            nper=periods,
            pv=-pv,
            fv=-fv,
            when=when_numeric
        )
        return {
            "payment": pmt,
            "pv": pv,
            "fv": fv,
            "rate": rate,
            "periods": periods
        }
    
    @staticmethod
    def calculate_amortization_schedule(
        principal: float,
        rate: float,
        periods: float,
        system: AmortizationSystem = AmortizationSystem.FRENCH
    ) -> list[dict[str, float]]:
        """
        Genera tabla de amortización.
        
        Args:
            principal: Capital inicial
            rate: Tasa de interés por período (decimal)
            periods: Número de períodos
            system: Sistema de amortización
            
        Returns:
            Lista de dicts con período, pago, interés, amortización, saldo
        """
        schedule = []
        balance = principal
        
        if system == AmortizationSystem.FRENCH:
            # Cuota fija
            pmt = npf.pmt(rate=rate, nper=periods, pv=-principal)
            for period in range(1, int(periods) + 1):
                interest = balance * rate
                amortization = pmt - interest
                balance -= amortization
                schedule.append({
                    "period": period,
                    "payment": pmt,
                    "interest": interest,
                    "amortization": amortization,
                    "balance": max(0, balance)  # Evitar negativos por redondeo
                })
        
        elif system == AmortizationSystem.GERMAN:
            # Amortización fija
            amortization = principal / periods
            for period in range(1, int(periods) + 1):
                interest = balance * rate
                pmt = amortization + interest
                balance -= amortization
                schedule.append({
                    "period": period,
                    "payment": pmt,
                    "interest": interest,
                    "amortization": amortization,
                    "balance": max(0, balance)
                })
        
        elif system == AmortizationSystem.AMERICAN:
            # Interés periódico + capital al final
            interest = principal * rate
            for period in range(1, int(periods) + 1):
                if period < periods:
                    schedule.append({
                        "period": period,
                        "payment": interest,
                        "interest": interest,
                        "amortization": 0,
                        "balance": principal
                    })
                else:
                    schedule.append({
                        "period": period,
                        "payment": principal + interest,
                        "interest": interest,
                        "amortization": principal,
                        "balance": 0
                    })
        
        return schedule
    
    @staticmethod
    def calculate_depreciation_schedule(
        cost: float,
        residual_value: float,
        useful_life: float,
        method: DepreciationMethod = DepreciationMethod.STRAIGHT_LINE
    ) -> list[dict[str, float]]:
        """
        Genera tabla de depreciación.
        
        Args:
            cost: Costo original del activo
            residual_value: Valor residual al final de la vida útil
            useful_life: Vida útil en períodos
            method: Método de depreciación
            
        Returns:
            Lista de dicts con período, depreciación, depreciación acumulada, valor neto
        """
        schedule = []
        depreciable_amount = cost - residual_value
        
        if method == DepreciationMethod.STRAIGHT_LINE:
            # Línea recta
            annual_depreciation = depreciable_amount / useful_life
            accumulated = 0
            for period in range(1, int(useful_life) + 1):
                accumulated += annual_depreciation
                net_value = cost - accumulated
                schedule.append({
                    "period": period,
                    "depreciation": annual_depreciation,
                    "accumulated_depreciation": accumulated,
                    "net_value": net_value
                })
        
        elif method == DepreciationMethod.DOUBLE_DECLINING:
            # Doble saldo decreciente
            rate = 2 / useful_life
            accumulated = 0
            book_value = cost
            for period in range(1, int(useful_life) + 1):
                depreciation = book_value * rate
                accumulated += depreciation
                book_value -= depreciation
                schedule.append({
                    "period": period,
                    "depreciation": depreciation,
                    "accumulated_depreciation": accumulated,
                    "net_value": book_value
                })
        
        elif method == DepreciationMethod.SUM_OF_DIGITS:
            # Suma de dígitos
            sum_of_years = sum(range(1, int(useful_life) + 1))
            accumulated = 0
            for period in range(1, int(useful_life) + 1):
                remaining_years = useful_life - period + 1
                depreciation = depreciable_amount * (remaining_years / sum_of_years)
                accumulated += depreciation
                net_value = cost - accumulated
                schedule.append({
                    "period": period,
                    "depreciation": depreciation,
                    "accumulated_depreciation": accumulated,
                    "net_value": net_value
                })
        
        return schedule
    
    @staticmethod
    def convert_interest_rate(
        rate: float,
        from_period: str,
        to_period: str
    ) -> float:
        """
        Convierte una tasa de interés entre períodos.
        
        Args:
            rate: Tasa a convertir (decimal)
            from_period: "annual", "monthly", "quarterly", etc.
            to_period: Período destino
            
        Returns:
            Tasa convertida
        """
        # Mapa de conversión a días
        period_days = {
            "annual": 365,
            "semi_annual": 182.5,
            "quarterly": 91.25,
            "monthly": 30.42,
            "weekly": 7,
            "daily": 1
        }
        
        from_days = period_days.get(from_period, 365)
        to_days = period_days.get(to_period, 365)
        
        # Usar fórmula de tasa equivalente
        converted_rate = (1 + rate) ** (to_days / from_days) - 1
        return converted_rate
    
    @staticmethod
    def calculate_irr(cashflows: list[float]) -> float:
        """
        Calcula la tasa interna de retorno (IRR).
        
        Args:
            cashflows: Lista de flujos de caja (con signo)
            
        Returns:
            IRR como decimal
        """
        try:
            return float(npf.irr(np.array(cashflows)))
        except:
            # Fallback a scipy.optimize si npf falla
            def npv_func(rate):
                return sum(cf / (1 + rate) ** i for i, cf in enumerate(cashflows))
            
            try:
                result = optimize.brentq(npv_func, -0.99, 10)
                return float(result)
            except:
                return 0.0
    
    @staticmethod
    def calculate_npv(rate: float, cashflows: list[float]) -> float:
        """
        Calcula el valor presente neto (NPV/VAN).
        
        Args:
            rate: Tasa de descuento (decimal)
            cashflows: Lista de flujos de caja
            
        Returns:
            NPV
        """
        return float(npf.npv(rate, np.array(cashflows)))
