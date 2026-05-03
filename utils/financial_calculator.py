"""
Financial Calculator Module
Handles all present value, future value, IRR, and comparison calculations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CalculationResult:
    """Result of a financial calculation."""
    metric_name: str
    value: float
    unit: str = ""
    description: str = ""


class FinancialCalculator:
    """Main financial calculator for loan/scenario comparisons."""

    @staticmethod
    def calculate_present_value(
        future_payment: float,
        annual_rate: float,
        months: int
    ) -> float:
        """
        Calculate present value of a future payment.
        PV = FV / (1 + r)^n
        """
        if annual_rate == 0:
            return future_payment
        
        monthly_rate = annual_rate / 100 / 12
        pv = future_payment / ((1 + monthly_rate) ** months)
        return round(pv, 2)

    @staticmethod
    def calculate_future_value(
        principal: float,
        annual_rate: float,
        months: int,
        monthly_payment: float = 0
    ) -> float:
        """
        Calculate future value of an investment/loan.
        FV = PV * (1 + r)^n + PMT * [((1 + r)^n - 1) / r]
        """
        if annual_rate == 0:
            # With zero interest, the future value is simply the total paid (payments sum).
            # If monthly payments exist, return total payments; otherwise return principal.
            if monthly_payment and monthly_payment > 0:
                return round(monthly_payment * months, 2)
            return round(principal, 2)
        
        monthly_rate = annual_rate / 100 / 12
        
        # Future value of principal
        fv_principal = principal * ((1 + monthly_rate) ** months)
        
        # Future value of annuity (monthly payments)
        if monthly_payment > 0:
            fv_annuity = monthly_payment * (((1 + monthly_rate) ** months - 1) / monthly_rate)
        else:
            fv_annuity = 0
        
        fv = fv_principal + fv_annuity
        return round(fv, 2)

    @staticmethod
    def calculate_total_interest(
        principal: float,
        monthly_payment: float,
        months: int
    ) -> float:
        """Calculate total interest paid over loan life."""
        total_paid = monthly_payment * months
        total_interest = total_paid - principal
        return round(max(0, total_interest), 2)

    @staticmethod
    def calculate_implied_rate(
        principal: float,
        monthly_payment: float,
        months: int,
        max_iterations: int = 100,
        tolerance: float = 0.0001
    ) -> Optional[float]:
        """
        Calculate implied annual interest rate using improved Newton-Raphson method.
        Given: principal, monthly payment, months
        Find: annual rate that satisfies: PV = PMT * [(1 - (1+r)^-n) / r]
        """
        if principal <= 0 or monthly_payment <= 0 or months <= 0:
            return None
        
        # Check if payment is too low (would require negative rate)
        min_total_payment = principal  # At least recover principal
        if monthly_payment * months < principal:
            return None
        
        # Helper function to calculate PV given monthly rate
        def pv_at_rate(r: float) -> float:
            if r == 0:
                return monthly_payment * months
            return monthly_payment * (1 - (1 + r) ** (-months)) / r
        
        # Helper function for derivative
        def pv_derivative(r: float) -> float:
            if r == 0:
                return -monthly_payment * months * (months + 1) / 2
            term1 = (1 + r) ** (-months - 1)
            return monthly_payment * (
                months * (1 + r) ** (-months - 1) * r -
                (1 - (1 + r) ** (-months))
            ) / (r * r)
        
        # Newton-Raphson iteration
        rate = 0.01  # Start with 1% monthly (~12% annual)
        
        for iteration in range(max_iterations):
            pv = pv_at_rate(rate)
            error = pv - principal
            
            # If we're close enough, return
            if abs(error) < tolerance:
                annual_rate = rate * 12 * 100
                return round(annual_rate, 2)
            
            # Calculate derivative
            deriv = pv_derivative(rate)
            
            if abs(deriv) < 1e-12:
                break
            
            # Newton step
            rate_new = rate - error / deriv
            
            # Ensure rate stays positive and reasonable
            if rate_new < 0:
                rate_new = rate / 2
            if rate_new > 1:  # Cap at 100% monthly
                rate_new = 1
            
            # Check convergence
            if abs(rate_new - rate) < tolerance / 100:
                annual_rate = rate_new * 12 * 100
                return round(annual_rate, 2)
            
            rate = rate_new
        
        # If Newton-Raphson fails, try bisection as fallback
        return FinancialCalculator._bisection_implied_rate(principal, monthly_payment, months)
    
    @staticmethod
    def _bisection_implied_rate(
        principal: float,
        monthly_payment: float,
        months: int
    ) -> Optional[float]:
        """
        Fallback bisection method for finding implied rate.
        More robust but slower than Newton-Raphson.
        """
        def pv_at_rate(r: float) -> float:
            if r == 0:
                return monthly_payment * months
            return monthly_payment * (1 - (1 + r) ** (-months)) / r
        
        # Search bounds: 0.001% to 100% monthly
        low = 0.00001
        high = 1.0
        
        # Check if solution exists in bounds
        pv_low = pv_at_rate(low) - principal
        pv_high = pv_at_rate(high) - principal
        
        if pv_low * pv_high > 0:
            return None  # No solution in bounds
        
        # Bisection
        for _ in range(50):  # 50 iterations gives ~15 decimal places precision
            mid = (low + high) / 2
            pv_mid = pv_at_rate(mid) - principal
            
            if abs(pv_mid) < 1e-6:
                annual_rate = mid * 12 * 100
                return round(annual_rate, 2)
            
            if pv_mid < 0:
                low = mid
            else:
                high = mid
        
        mid = (low + high) / 2
        annual_rate = mid * 12 * 100
        return round(annual_rate, 2)

    @staticmethod
    def generate_amortization_schedule(
        principal: float,
        annual_rate: float,
        months: int,
        monthly_payment: Optional[float] = None
    ) -> List[Dict]:
        """
        Generate amortization schedule (month-by-month breakdown).
        Returns list of dicts with: month, payment, principal, interest, balance
        """
        schedule = []
        
        if annual_rate == 0:
            monthly_rate = 0
        else:
            monthly_rate = annual_rate / 100 / 12
        
        # Calculate payment if not provided
        if monthly_payment is None or monthly_payment == 0:
            if annual_rate == 0:
                monthly_payment = principal / months
            else:
                monthly_payment = principal * (monthly_rate * (1 + monthly_rate) ** months) / \
                                 ((1 + monthly_rate) ** months - 1)
        
        balance = principal
        total_interest = 0
        
        for month in range(1, months + 1):
            interest = balance * monthly_rate
            principal_payment = monthly_payment - interest
            balance -= principal_payment
            total_interest += interest
            
            # Avoid negative balance in last month
            if month == months and balance < 0:
                principal_payment += balance
                balance = 0
            
            schedule.append({
                "month": month,
                "payment": round(monthly_payment, 2),
                "principal": round(principal_payment, 2),
                "interest": round(interest, 2),
                "balance": round(max(0, balance), 2)
            })
        
        return schedule

    @staticmethod
    def compare_scenarios(
        scenarios: List[Dict],
        comparison_metric: str = "present_value",
        discount_rate: float = 12,
        time_horizon_months: int = None
    ) -> Dict:
        """
        Compare multiple loan/scenario options.
        
        Args:
            scenarios: List of dicts with fields: name, principal, annual_rate, months, monthly_payment
            comparison_metric: "present_value", "future_value", "total_interest", "monthly_payment"
            discount_rate: Rate for PV calculation (%)
            time_horizon_months: For VF calculation, if None use loan months
        
        Returns:
            Dict with comparison results and recommendation
        """
        results = {}
        
        for scenario in scenarios:
            name = scenario.get("name", "Unnamed")
            principal = scenario.get("principal", 0)
            annual_rate = scenario.get("annual_rate", 0)
            months = scenario.get("months", 0)
            monthly_payment = scenario.get("monthly_payment", 0)
            
            # Calculate metrics
            if comparison_metric == "present_value":
                if time_horizon_months is None:
                    time_horizon_months = months
                
                # PV of future payments
                schedule = FinancialCalculator.generate_amortization_schedule(
                    principal, annual_rate, months, monthly_payment
                )
                pv = principal  # PV of principal is itself
                for entry in schedule[:time_horizon_months]:
                    pv += FinancialCalculator.calculate_present_value(
                        entry["payment"], discount_rate, entry["month"]
                    )
                value = pv
                unit = "$"
                description = f"Present Value over {time_horizon_months} months @ {discount_rate}% discount"
            
            elif comparison_metric == "future_value":
                if time_horizon_months is None:
                    time_horizon_months = months
                
                value = FinancialCalculator.calculate_future_value(
                    principal, annual_rate, time_horizon_months, monthly_payment
                )
                unit = "$"
                description = f"Future Value over {time_horizon_months} months"
            
            elif comparison_metric == "total_interest":
                value = FinancialCalculator.calculate_total_interest(
                    principal, monthly_payment, months
                )
                unit = "$"
                description = "Total Interest Paid"
            
            elif comparison_metric == "monthly_payment":
                value = monthly_payment
                unit = "$/month"
                description = "Monthly Payment"
            
            else:
                value = 0
                unit = ""
                description = "Unknown metric"
            
            results[name] = {
                "value": round(value, 2),
                "unit": unit,
                "description": description
            }
        
        # Find best option (lowest for all metrics except monthly_payment which could be interpreted either way)
        if comparison_metric in ["total_interest", "monthly_payment"]:
            best = min(results.items(), key=lambda x: x[1]["value"])
            recommendation = f"✅ {best[0]} is better (lower {comparison_metric})"
        else:
            best = min(results.items(), key=lambda x: x[1]["value"])
            recommendation = f"✅ {best[0]} is better"
        
        return {
            "metric": comparison_metric,
            "results": results,
            "best_option": best[0],
            "recommendation": recommendation
        }

    @staticmethod
    def suggest_comparison_metrics(loan_data: Dict) -> List[str]:
        """
        Based on loan data, suggest which metrics to compare.
        """
        suggestions = []
        
        principal = loan_data.get("principal", 0)
        annual_rate = loan_data.get("annual_rate", 0)
        months = loan_data.get("months", 0)
        monthly_payment = loan_data.get("monthly_payment", 0)
        
        # Always suggest these if we have principal and at least one of: rate or payment
        if principal > 0:
            if (annual_rate > 0 or monthly_payment > 0) and months > 0:
                suggestions.append("total_interest")  # Total interest paid
                suggestions.append("future_value")    # Future value of payments
        
        if monthly_payment > 0:
            suggestions.append("monthly_payment")     # Monthly payment comparison
        
        if annual_rate > 0:
            suggestions.append("present_value")       # PV for financial analysis
        
        # Default if nothing else
        if not suggestions:
            suggestions.append("total_interest")
        
        return suggestions

    @staticmethod
    def calculate_all_metrics(scenario: Dict) -> Dict:
        """
        Calculate all relevant metrics for a single scenario.
        Returns dict with all calculated values.
        """
        principal = scenario.get("principal", 0)
        annual_rate = scenario.get("annual_rate", 0)
        months = scenario.get("months", 0)
        monthly_payment = scenario.get("monthly_payment", 0)
        
        results = {}
        
        # Calculate implied rate if missing
        if annual_rate == 0 and principal > 0 and monthly_payment > 0 and months > 0:
            implied_rate = FinancialCalculator.calculate_implied_rate(
                principal, monthly_payment, months
            )
            if implied_rate:
                annual_rate = implied_rate
        
        # Generate amortization (use first 12 months for summary)
        if principal > 0:
            schedule = FinancialCalculator.generate_amortization_schedule(
                principal, annual_rate, months, monthly_payment
            )
            results["schedule"] = schedule
            
            # Summary metrics
            results["total_interest_paid"] = sum(s["interest"] for s in schedule)
            results["total_principal_paid"] = sum(s["principal"] for s in schedule)
            results["implied_annual_rate"] = annual_rate
            
            # PV and FV
            results["present_value"] = principal
            results["future_value"] = FinancialCalculator.calculate_future_value(
                principal, annual_rate, months, monthly_payment
            )
        
        return results
