import json

from utils.merchant_database import MerchantDatabase


def test_international_transfer_uses_transaction_direction():
    database = MerchantDatabase()

    incoming = database.categorize_transaction(
        "Incoming SWIFT international transfer", is_debit=False
    )
    outgoing = database.categorize_transaction(
        "Outgoing SWIFT international transfer", is_debit=True
    )

    assert incoming == ("international_transfer_in", 0.95)
    assert outgoing == ("international_transfer_out", 0.95)


def test_international_transfer_requires_known_direction():
    database = MerchantDatabase()

    category, _ = database.categorize_transaction("International wire transfer")

    assert category == "uncategorized"


def test_other_income_requires_credit_direction():
    database = MerchantDatabase()

    credit = database.categorize_transaction("Freelance income July", is_debit=False)
    debit = database.categorize_transaction("Freelance income course", is_debit=True)

    assert credit == ("other_income", 0.90)
    assert debit[0] != "other_income"


def test_existing_keyword_category_is_preserved():
    database = MerchantDatabase()

    assert database.categorize_transaction(
        "STARBUCKS STORE 123", is_debit=True
    )[0] == "food_dining"


def test_custom_categories_are_never_saved_as_merchant_rules(tmp_path):
    database = MerchantDatabase()
    database.user_rules_path = str(tmp_path / "rules.json")

    saved = database.save_user_category_rule(
        "Neighborhood Pharmacy", "custom_aux_1"
    )

    assert saved is False
    assert "neighborhood pharmacy" not in database.user_category_overrides
    assert not (tmp_path / "rules.json").exists()

    assert database.save_user_category_rule(
        "Future custom slot", "custom_aux_99"
    ) is False


def test_custom_rules_are_ignored_if_found_in_a_rules_file(tmp_path):
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "normal merchant": "groceries",
                "personal merchant": "custom_aux_1",
            }
        ),
        encoding="utf-8",
    )
    database = MerchantDatabase()
    database.user_rules_path = str(rules_path)

    loaded = database._load_user_category_overrides()

    assert loaded == {"normal merchant": "groceries"}


def test_standard_financial_category_still_persists(tmp_path):
    database = MerchantDatabase()
    database.user_rules_path = str(tmp_path / "rules.json")
    database.user_category_overrides = {}

    assert database.save_user_category_rule(
        "Neighborhood Pharmacy", "healthcare"
    ) is True

    saved = json.loads((tmp_path / "rules.json").read_text(encoding="utf-8"))
    assert saved == {"neighborhood pharmacy": "healthcare"}
