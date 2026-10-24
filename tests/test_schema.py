from lc_risk import schema

# Post-origination and pricing fields that must never become model inputs.
FORBIDDEN_INPUTS = {
    *schema.PRICING_COLS,
    *schema.OUTCOME_COLS,
    "grade",
    "total_rec_prncp",
    "total_rec_int",
    "last_fico_range_low",
    "last_fico_range_high",
    "last_credit_pull_d",
    "debt_settlement_flag",
    "hardship_flag",
}


def test_field_names_are_unique():
    names = [f.name for f in schema.ALL_INPUT_FIELDS]
    assert len(names) == len(set(names))


def test_no_leakage_or_pricing_in_inputs():
    names = {f.name for f in schema.ALL_INPUT_FIELDS}
    assert not names & FORBIDDEN_INPUTS


def test_model_a_and_b_differ_only_by_lc_rating():
    a = {f.name for f in schema.model_inputs(include_lc_rating=False)}
    b = {f.name for f in schema.model_inputs(include_lc_rating=True)}
    assert b - a == {"sub_grade"}
    assert a < b


def test_categorical_fields_declare_categories():
    for field in schema.ALL_INPUT_FIELDS:
        if field.kind == "categorical":
            assert field.categories, field.name
            assert len(field.categories) == len(set(field.categories)), field.name


def test_excluded_fields_are_accepted_but_not_model_inputs():
    accepted = {f.name for f in schema.ALL_INPUT_FIELDS}
    for include in (False, True):
        used = {f.name for f in schema.model_inputs(include_lc_rating=include)}
        assert set(schema.EXCLUDED_FROM_MODEL) <= accepted
        assert not set(schema.EXCLUDED_FROM_MODEL) & used


def test_event_history_fields_are_nullable_and_core_fields_are_not():
    by_name = schema.FIELDS_BY_NAME
    assert all(by_name[n].nullable for n in by_name if n.startswith("mths_since_"))
    assert not any(by_name[n].nullable for n in ("loan_amnt", "annual_inc", "fico_range_low"))
