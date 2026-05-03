import math
from utils.financial_solver import implied_rate_from_payment, loan_payment, amortization_french


def test_implied_rate_roundtrip():
    principal = 20000.0
    # annual effective 5%
    effective_annual = 0.05
    periodic = (1 + effective_annual) ** (1 / 12) - 1
    payment, _ = amortization_french(principal, periodic, 60)

    solved = implied_rate_from_payment(principal, payment, 60, method="french")
    assert abs(solved["effective_annual"] - effective_annual) < 1e-6


def test_loan_payment_infers_rate_from_monthly():
    principal = 15000.0
    periods = 36
    # create a loan with known rate and get its payment
    effective_annual = 0.12
    periodic = (1 + effective_annual) ** (1 / 12) - 1
    payment, _ = amortization_french(principal, periodic, periods)

    # Now call loan_payment with missing rate but with monthly_payment provided
    payload = {
        "principal": principal,
        "periods": periods,
        "monthly_payment": payment,
        "method": "french",
    }
    result = loan_payment(payload)
    # The solver should infer a periodic_rate_used close to periodic
    periodic_used = result["inputs"]["periodic_rate_used"]
    assert abs(periodic_used - periodic) < 1e-8
