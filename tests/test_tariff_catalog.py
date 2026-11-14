from arogya.lib.tariff_db import DATA, lookup_benchmark


def test_catalog_has_thirty_named_procedures_with_all_city_tiers():
    assert len(DATA) >= 30
    names = [entry["procedureName"].strip().casefold() for entry in DATA]
    assert len(names) == len(set(names))
    for entry in DATA:
        assert set(entry["benchmarks"]) == {"tier1", "tier2", "tier3"}
        for benchmark in entry["benchmarks"].values():
            assert 0 < benchmark["min"] <= benchmark["standardAvg"] <= benchmark["max"]


def test_added_procedure_can_be_matched_by_tariff_lookup():
    result = lookup_benchmark("CABG", "tier2")
    assert result["found"] is True
    assert result["min"] <= result["standardAvg"] <= result["max"]
