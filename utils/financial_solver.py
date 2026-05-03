import math
from datetime import datetime
from typing import Dict, List, Tuple


def _round2(value: float) -> float:
    return round(float(value), 2)


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _convert_rate_to_period(rate_value: float, rate_type: str, periods_per_year: int = 12) -> float:
    rate = _safe_float(rate_value)
    rate_type = (rate_type or "effective_annual").lower()

    if rate_type == "effective_annual":
        return (1 + rate) ** (1 / periods_per_year) - 1
    if rate_type == "nominal_annual":
        return rate / periods_per_year
    if rate_type == "effective_monthly":
        return rate

    return (1 + rate) ** (1 / periods_per_year) - 1


def present_value(future_value: float, rate: float, periods: int) -> Dict:
    pv = _safe_float(future_value) / ((1 + _safe_float(rate)) ** int(periods))
    return {
        "type": "present_value",
        "inputs": {
            "future_value": _safe_float(future_value),
            "rate": _safe_float(rate),
            "periods": int(periods),
        },
        "formula": "PV = FV / (1+r)^n",
        "result": {
            "present_value": _round2(pv),
        },
    }


def future_value(present_value_amount: float, rate: float, periods: int) -> Dict:
    fv = _safe_float(present_value_amount) * ((1 + _safe_float(rate)) ** int(periods))
    return {
        "type": "future_value",
        "inputs": {
            "present_value": _safe_float(present_value_amount),
            "rate": _safe_float(rate),
            "periods": int(periods),
        },
        "formula": "FV = PV * (1+r)^n",
        "result": {
            "future_value": _round2(fv),
        },
    }


def amortization_french(principal: float, periodic_rate: float, periods: int) -> Tuple[float, List[Dict]]:
    p = _safe_float(principal)
    r = _safe_float(periodic_rate)
    n = int(periods)

    if n <= 0:
        return 0.0, []

    if r == 0:
        payment = p / n
    else:
        payment = p * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

    balance = p
    schedule = []
    for i in range(1, n + 1):
        interest = balance * r
        principal_paid = payment - interest
        if i == n:
            principal_paid = balance
            payment = principal_paid + interest
        balance -= principal_paid
        schedule.append(
            {
                "period": i,
                "payment": _round2(payment),
                "interest": _round2(interest),
                "principal": _round2(principal_paid),
                "balance": _round2(max(balance, 0.0)),
            }
        )
    return payment, schedule


def amortization_german(principal: float, periodic_rate: float, periods: int) -> Tuple[List[Dict], float]:
    p = _safe_float(principal)
    r = _safe_float(periodic_rate)
    n = int(periods)

    if n <= 0:
        return [], 0.0

    principal_part = p / n
    balance = p
    schedule = []
    total_paid = 0.0

    for i in range(1, n + 1):
        interest = balance * r
        payment = principal_part + interest
        if i == n:
            principal_part = balance
            payment = principal_part + interest
        balance -= principal_part
        total_paid += payment
        schedule.append(
            {
                "period": i,
                "payment": _round2(payment),
                "interest": _round2(interest),
                "principal": _round2(principal_part),
                "balance": _round2(max(balance, 0.0)),
            }
        )

    return schedule, total_paid


def amortization_american(principal: float, periodic_rate: float, periods: int) -> Tuple[List[Dict], float]:
    p = _safe_float(principal)
    r = _safe_float(periodic_rate)
    n = int(periods)

    if n <= 0:
        return [], 0.0

    schedule = []
    total_paid = 0.0

    for i in range(1, n + 1):
        interest = p * r
        principal_paid = 0.0
        payment = interest
        if i == n:
            principal_paid = p
            payment = p + interest
        total_paid += payment
        schedule.append(
            {
                "period": i,
                "payment": _round2(payment),
                "interest": _round2(interest),
                "principal": _round2(principal_paid),
                "balance": _round2(p if i < n else 0.0),
            }
        )

    return schedule, total_paid


def implied_rate_from_payment(principal: float, payment: float, periods: int, method: str = "french", tol: float = 1e-9, max_iter: int = 200) -> Dict:
    """Solve for periodic rate r such that the amortization payment equals the provided payment.

    Returns a dict with `periodic_rate`, `effective_annual` and `nominal_annual`.
    Uses bisection to guarantee convergence and robustness for typical consumer rates.
    """
    p = _safe_float(principal)
    target = _safe_float(payment)
    n = int(periods)

    if n <= 0:
        raise ValueError("periods must be > 0")
    if target <= 0:
        raise ValueError("payment must be > 0")

    def payment_for_r(r):
        if method == "french":
            pay, _ = amortization_french(p, r, n)
            return pay
        elif method == "german":
            schedule, _ = amortization_german(p, r, n)
            return schedule[0]["payment"] if schedule else 0.0
        else:
            schedule, _ = amortization_american(p, r, n)
            return schedule[0]["payment"] if schedule else 0.0

    # bracket search: start from near-zero up to a large rate
    low = 0.0
    high = 1.0

    # expand upper bound until signs differ or we reach very large rate
    for _ in range(80):
        f_low = payment_for_r(low) - target
        f_high = payment_for_r(high) - target
        if f_low == 0:
            r = low
            break
        if f_low * f_high < 0:
            break
        high *= 2.0
    else:
        # fallback: if we couldn't bracket, try a safe guess
        r = 0.0

    # If r wasn't set in the bracket loop, perform bisection
    if 'r' not in locals():
        for i in range(max_iter):
            mid = (low + high) / 2.0
            f_mid = payment_for_r(mid) - target
            if abs(f_mid) < tol:
                r = mid
                break
            if (payment_for_r(low) - target) * f_mid < 0:
                high = mid
            else:
                low = mid
        else:
            r = mid

    effective_annual = (1 + r) ** 12 - 1
    nominal_annual = r * 12
    return {"periodic_rate": r, "effective_annual": effective_annual, "nominal_annual": nominal_annual}


def loan_payment(problem: Dict) -> Dict:
    principal = _safe_float(problem.get("principal"))
    periods = int(problem.get("periods", 0))
    rate_value = _safe_float(problem.get("rate"))
    rate_type = problem.get("rate_type", "effective_annual")
    method = (problem.get("method", "french") or "french").lower()

    # If rate is missing but a monthly payment or total obligation is provided,
    # attempt to infer the implicit periodic rate that matches the payment.
    monthly_payment_input = problem.get("monthly_payment")
    total_obligation = problem.get("total_obligation")

    if rate_value in [None, "", 0, "0"] and (monthly_payment_input or total_obligation):
        # Determine an indicative monthly payment
        if monthly_payment_input not in [None, "", 0, "0"]:
            payment = _safe_float(monthly_payment_input)
        else:
            # If total_obligation provided, assume equal installments
            periods_for_calc = periods if periods > 0 else int(problem.get("periods", 0))
            payment = _safe_float(total_obligation) / periods_for_calc if periods_for_calc > 0 else 0.0

        try:
            implied = None
            # lazy import of helper function (defined below)
            implied = implied_rate_from_payment(principal, payment, periods, method=method)
            # implied contains periodic_rate and annualized rates
            r = implied.get("periodic_rate", 0.0)
            # set a sensible rate_value (effective annual) so other code has it
            rate_value = implied.get("effective_annual", 0.0)
            rate_type = "effective_annual"
        except Exception:
            # fallback to normal conversion when inference fails
            r = _convert_rate_to_period(rate_value, rate_type)
    else:
        r = _convert_rate_to_period(rate_value, rate_type)

    if method == "french":
        payment, schedule = amortization_french(principal, r, periods)
        total_paid = sum(item["payment"] for item in schedule)
    elif method == "german":
        schedule, total_paid = amortization_german(principal, r, periods)
        payment = schedule[0]["payment"] if schedule else 0.0
    else:
        schedule, total_paid = amortization_american(principal, r, periods)
        payment = schedule[0]["payment"] if schedule else 0.0

    total_interest = total_paid - principal

    return {
        "type": "loan_payment",
        "inputs": {
            "principal": principal,
            "periods": periods,
            "rate": rate_value,
            "rate_type": rate_type,
            "method": method,
            "periodic_rate_used": r,
        },
        "formula": "Depends on method: French/German/American amortization",
        "result": {
            "estimated_payment": _round2(payment),
            "total_paid": _round2(total_paid),
            "total_interest": _round2(total_interest),
            "schedule_preview": schedule[:12],
        },
    }


def compare_loans(problem: Dict) -> Dict:
    loans = problem.get("loans", [])
    results = []

    for loan in loans:
        one = loan_payment(loan)
        results.append(
            {
                "name": loan.get("name", "Loan"),
                "method": loan.get("method", "french"),
                "estimated_payment": one["result"]["estimated_payment"],
                "total_paid": one["result"]["total_paid"],
                "total_interest": one["result"]["total_interest"],
            }
        )

    winner = None
    if results:
        winner = min(results, key=lambda x: x["total_paid"])

    return {
        "type": "compare_loans",
        "inputs": {
            "loans": loans,
        },
        "formula": "Compare total paid and total interest across alternatives",
        "result": {
            "comparison": results,
            "best_by_total_paid": winner,
        },
    }


def compare_asset_options(problem: Dict) -> Dict:
    inputs = problem.get("inputs", {}) or {}
    options = problem.get("options", inputs.get("options", [])) or []
    annual_discount_rate = _safe_float(problem.get("discount_rate", inputs.get("discount_rate", 0.0)))
    horizon_months = int(problem.get("horizon_months", inputs.get("horizon_months", 48)))
    periodic_discount_rate = _convert_rate_to_period(annual_discount_rate, problem.get("rate_type", inputs.get("rate_type", "effective_annual")))

    results = []
    for option in options:
        name = option.get("name", "Opción")
        option_type = (option.get("option_type") or "purchase").lower().strip()
        upfront_payment = _safe_float(option.get("upfront_payment", option.get("purchase_price", 0.0)))
        monthly_payment = _safe_float(option.get("monthly_payment", 0.0))
        term_months = int(option.get("term_months", horizon_months) or horizon_months)
        residual_value = _safe_float(option.get("residual_value", 0.0))
        maintenance_monthly = _safe_float(option.get("maintenance_monthly", 0.0))
        maintenance_included = bool(option.get("maintenance_included", False))
        lease_purchase_enabled = bool(option.get("lease_purchase_enabled", False))
        lease_purchase_price = _safe_float(option.get("lease_purchase_price", 0.0))
        residual_timing_months = int(option.get("residual_timing_months", term_months or horizon_months))

        pv_cost = upfront_payment
        for month in range(1, horizon_months + 1):
            discount_factor = (1 + periodic_discount_rate) ** month

            if monthly_payment > 0 and (term_months <= 0 or month <= term_months):
                pv_cost += monthly_payment / discount_factor

            if maintenance_monthly > 0 and not maintenance_included:
                pv_cost += maintenance_monthly / discount_factor

        if residual_value > 0 and residual_timing_months > 0 and residual_timing_months <= horizon_months:
            discount_factor = (1 + periodic_discount_rate) ** residual_timing_months
            if option_type in ["purchase", "buy", "compra", "credit", "credito"]:
                pv_cost -= residual_value / discount_factor
            elif option_type in ["leasing", "lease", "rental_purchase"]:
                if lease_purchase_enabled:
                    pv_cost += lease_purchase_price / discount_factor

        if periodic_discount_rate == 0:
            equivalent_monthly_cost = pv_cost / horizon_months if horizon_months > 0 else pv_cost
        else:
            annuity_factor = (periodic_discount_rate / (1 - (1 + periodic_discount_rate) ** (-horizon_months))) if horizon_months > 0 else 1.0
            equivalent_monthly_cost = pv_cost * annuity_factor if annuity_factor else pv_cost

        results.append(
            {
                "name": name,
                "option_type": option_type,
                "horizon_months": horizon_months,
                "upfront_payment": _round2(upfront_payment),
                "monthly_payment": _round2(monthly_payment),
                "term_months": term_months,
                "residual_value": _round2(residual_value),
                "maintenance_monthly": _round2(maintenance_monthly),
                "maintenance_included": maintenance_included,
                "lease_purchase_enabled": lease_purchase_enabled,
                "lease_purchase_price": _round2(lease_purchase_price),
                "residual_timing_months": residual_timing_months,
                "present_cost": _round2(pv_cost),
                "equivalent_monthly_cost": _round2(equivalent_monthly_cost),
            }
        )

    winner = None
    if results:
        winner = min(results, key=lambda x: x["present_cost"])

    return {
        "type": "compare_asset_options",
        "inputs": {
            "discount_rate": annual_discount_rate,
            "rate_type": problem.get("rate_type", "effective_annual"),
            "horizon_months": horizon_months,
            "options": options,
        },
        "formula": "Compare present value of total ownership / leasing / rental cost over a common horizon",
        "result": {
            "comparison": results,
            "best_by_present_cost": winner,
        },
    }


def rate_conversion(problem: Dict) -> Dict:
    rate = _safe_float(problem.get("rate", 0.0))
    from_type = (problem.get("from_type", "effective_annual") or "effective_annual").lower()

    monthly = _convert_rate_to_period(rate, from_type)
    effective_annual = (1 + monthly) ** 12 - 1
    nominal_annual = monthly * 12

    return {
        "type": "rate_conversion",
        "inputs": {
            "rate": rate,
            "from_type": from_type,
            "periodicity": "monthly",
        },
        "formula": "Rate normalization to monthly, nominal annual, and effective annual",
        "result": {
            "effective_monthly": round(monthly, 6),
            "nominal_annual": round(nominal_annual, 6),
            "effective_annual": round(effective_annual, 6),
        },
    }


def refinance(problem: Dict) -> Dict:
    current = problem.get("current_loan", {})
    proposed = problem.get("proposed_loan", {})

    current_result = loan_payment(current)
    proposed_result = loan_payment(proposed)

    current_total = _safe_float(current_result["result"]["total_paid"])
    proposed_total = _safe_float(proposed_result["result"]["total_paid"])

    return {
        "type": "refinance",
        "inputs": {
            "current_loan": current,
            "proposed_loan": proposed,
        },
        "formula": "Compare current vs proposed total paid and monthly payment",
        "result": {
            "current_total_paid": _round2(current_total),
            "proposed_total_paid": _round2(proposed_total),
            "estimated_savings_total": _round2(current_total - proposed_total),
            "current_payment": current_result["result"]["estimated_payment"],
            "proposed_payment": proposed_result["result"]["estimated_payment"],
        },
    }


def npv_irr(problem: Dict) -> Dict:
    cashflows = [_safe_float(x) for x in problem.get("cashflows", [])]
    discount_rate = _safe_float(problem.get("discount_rate", 0.0))

    npv = 0.0
    for t, cf in enumerate(cashflows):
        npv += cf / ((1 + discount_rate) ** t)

    def irr_newton(flows: List[float], guess: float = 0.1, max_iter: int = 100, tol: float = 1e-7):
        r = guess
        for _ in range(max_iter):
            f = 0.0
            df = 0.0
            for t, cf in enumerate(flows):
                denom = (1 + r) ** t
                f += cf / denom
                if t > 0:
                    df -= t * cf / ((1 + r) ** (t + 1))
            if abs(df) < tol:
                break
            new_r = r - f / df
            if abs(new_r - r) < tol:
                return new_r
            r = new_r
        return r

    irr = None
    if len(cashflows) >= 2 and any(cf < 0 for cf in cashflows) and any(cf > 0 for cf in cashflows):
        irr = irr_newton(cashflows)

    return {
        "type": "npv_irr",
        "inputs": {
            "cashflows": cashflows,
            "discount_rate": discount_rate,
        },
        "formula": "NPV = Σ(CFt/(1+r)^t), IRR solved iteratively",
        "result": {
            "npv": _round2(npv),
            "irr": round(irr, 6) if irr is not None else None,
        },
    }


def real_return(problem: Dict) -> Dict:
    nominal_return = _safe_float(problem.get("nominal_return", 0.0))
    inflation = _safe_float(problem.get("inflation", 0.0))

    real = ((1 + nominal_return) / (1 + inflation)) - 1

    return {
        "type": "real_return",
        "inputs": {
            "nominal_return": nominal_return,
            "inflation": inflation,
        },
        "formula": "Real return = (1+nominal)/(1+inflation)-1",
        "result": {
            "real_return": round(real, 6),
        },
    }


def debt_structures(problem: Dict) -> Dict:
    structures = problem.get("structures", [])
    rows = []
    for s in structures:
        solved = loan_payment(s)
        rows.append(
            {
                "name": s.get("name", "Structure"),
                "payment": solved["result"]["estimated_payment"],
                "total_paid": solved["result"]["total_paid"],
                "total_interest": solved["result"]["total_interest"],
            }
        )

    best = min(rows, key=lambda x: x["total_paid"]) if rows else None

    return {
        "type": "debt_structures",
        "inputs": {
            "structures": structures,
        },
        "formula": "Compares cost and burden across debt structures",
        "result": {
            "comparison": rows,
            "best_structure": best,
        },
    }


def debt_payment_alternatives(problem: Dict) -> Dict:
    principal = _safe_float(problem.get("principal", 0))
    rate_value = _safe_float(problem.get("rate", 0))
    rate_type = problem.get("rate_type", "effective_annual")
    planned_periods = int(problem.get("periods", 0))
    extra_payment = _safe_float(problem.get("extra_payment", 0))

    base = loan_payment(
        {
            "principal": principal,
            "rate": rate_value,
            "rate_type": rate_type,
            "periods": planned_periods,
            "method": "french",
        }
    )

    periodic_rate = _convert_rate_to_period(rate_value, rate_type)
    base_payment = _safe_float(base["result"]["estimated_payment"])
    new_payment = base_payment + extra_payment

    # Simulate faster payoff with extra payment
    balance = principal
    months = 0
    total_paid = 0.0
    while balance > 1e-9 and months < 2000:
        months += 1
        interest = balance * periodic_rate
        principal_paid = max(new_payment - interest, 0.0)
        if principal_paid > balance:
            principal_paid = balance
        payment = principal_paid + interest
        balance -= principal_paid
        total_paid += payment

    return {
        "type": "debt_payment_alternatives",
        "inputs": {
            "principal": principal,
            "rate": rate_value,
            "rate_type": rate_type,
            "periods": planned_periods,
            "extra_payment": extra_payment,
        },
        "formula": "Baseline amortization vs accelerated payment with extra monthly amount",
        "result": {
            "baseline_total_paid": _safe_float(base["result"]["total_paid"]),
            "accelerated_total_paid": _round2(total_paid),
            "baseline_periods": planned_periods,
            "accelerated_periods": months,
            "total_savings": _round2(_safe_float(base["result"]["total_paid"]) - total_paid),
        },
    }


def solve_problem(problem: Dict) -> Dict:
    problem_type = (problem.get("problem_type") or "").lower().strip()

    solver_map = {
        "present_value": lambda p: present_value(p.get("future_value", 0), p.get("rate", 0), p.get("periods", 0)),
        "future_value": lambda p: future_value(p.get("present_value", 0), p.get("rate", 0), p.get("periods", 0)),
        "loan_payment": loan_payment,
        "compare_loans": compare_loans,
        "compare_asset_options": compare_asset_options,
        "rate_conversion": rate_conversion,
        "refinance": refinance,
        "npv_irr": npv_irr,
        "real_return": real_return,
        "debt_structures": debt_structures,
        "debt_payment_alternatives": debt_payment_alternatives,
    }

    if problem_type not in solver_map:
        raise ValueError(f"Unsupported problem_type: {problem_type}")

    solved = solver_map[problem_type](problem)

    return {
        "problem_type": problem_type,
        "timestamp": datetime.now().isoformat(),
        "trace": {
            "inputs": solved.get("inputs", {}),
            "assumptions": problem.get("assumptions", []),
            "formula": solved.get("formula", ""),
            "result": solved.get("result", {}),
        },
        "result": solved.get("result", {}),
    }
