from lc_risk import config


def test_split_years_are_disjoint_and_ordered():
    years = [y for split in ("train", "validation", "test") for y in config.SPLIT_YEARS[split]]
    assert years == sorted(years)
    assert len(years) == len(set(years))


def test_label_statuses_do_not_overlap():
    assert not config.NEGATIVE_STATUSES & config.POSITIVE_STATUSES


def test_source_digest_is_stable_and_hex():
    from lc_risk.provenance import source_digest

    first = source_digest()
    assert first == source_digest()
    assert len(first) == 64 and int(first, 16) >= 0
