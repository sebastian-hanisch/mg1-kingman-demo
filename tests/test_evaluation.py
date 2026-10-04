"""Auswertung: Live-Lauf, Zellen-Kennzahlen von Hand, Vollständigkeit der vorgerechneten Datei."""

import pytest

import mg1_constants as C
import mg1_evaluation as E
import mg1_formulas as F
from mg1_simulation import SimResult


def fake_run(wait, prob=0.5):
    return SimResult(c=1, rho=0.5, ca2=1.0, cs2=1.0, n=100, mean_wait=wait, wait_prob=prob, utilisation=0.5, mean_interarrival=2.0, mean_service=1.0)


def test_mean_and_sd_by_hand():
    assert E.mean_and_sd([1.0, 3.0]) == (2.0, pytest.approx(2 ** 0.5)) and E.mean_and_sd([5.0]) == (5.0, 0.0)


def test_formulas_collects_all_reference_values():
    f = E.formulas(1, 80, 1.0, 4.0)
    assert f["approx"] == pytest.approx(10.0) and f["exact"] == pytest.approx(10.0) and f["mm"] == pytest.approx(4.0)
    assert E.formulas(4, 80, 0.0, 4.0)["exact"] is None and E.formulas(4, 80, 0.0, 4.0)["approx"] == pytest.approx(2.0 * F.mmc_wait(4, 0.8))


def test_cell_figures_by_hand():
    """Simulation 1.0 / 3.0 → Mittel 2.0, Standardfehler 1.0, relative Streuung √2/2; Näherung 2.2 → +10 %; exakt 2.0 → Simulation +0 %; Aufwand für ±5 %: 100 000·(0.7071/0.05)²."""
    cell = E.summarize_runs(1, 50, 1.0, 1.0, 100_000, [fake_run(1.0), fake_run(3.0)])
    cell.update({"approx": 2.2, "exact": 2.0})
    assert E.cell_mean(cell) == 2.0 and E.cell_se(cell) == pytest.approx(1.0) and E.cell_rel_sd(cell) == pytest.approx(0.5 * 2 ** 0.5)
    assert E.approx_error(cell) == pytest.approx(0.1) and E.exact_error(cell) == pytest.approx(0.0)
    assert E.customers_needed(cell) == pytest.approx(100_000 * (0.5 * 2 ** 0.5 / 0.05) ** 2)


def test_errors_are_none_without_waiting_or_without_a_formula():
    never = E.summarize_runs(1, 50, 0.0, 0.0, 1000, [fake_run(0.0), fake_run(0.0)])
    assert E.approx_error(never) is None and E.customers_needed(never) != E.customers_needed(never)         # nan
    no_formula = E.summarize_runs(4, 80, 0.0, 4.0, 1000, [fake_run(1.0), fake_run(1.2)])
    assert no_formula["exact"] is None and E.exact_error(no_formula) is None


def test_nearest_picks_the_closest_option_and_the_smaller_on_a_tie():
    assert E.nearest(C.STUDY_RHO_PCT, 60) == 50 and E.nearest(C.STUDY_RHO_PCT, 65) == 50 and E.nearest(C.STUDY_RHO_PCT, 66) == 80
    assert E.nearest(C.STUDY_RHO_PCT, 90) == 95 and E.nearest(C.STUDY_RHO_PCT, 87) == 80


def test_live_report_structure_and_reference_values():
    r = E.live_report(4, 80, 1.0, 1.0, seed=3, customers=8_000)
    assert r["exact"] == pytest.approx(F.mmc_wait(4, 0.8)) and r["approx"] == pytest.approx(F.mmc_wait(4, 0.8)) and r["mm"] == pytest.approx(F.mmc_wait(4, 0.8))
    assert r["wait_sim"] == r["sim"].mean_wait and r["sim"].n == 8_000 - int(0.05 * 8_000)


def test_study_cell_run_small_structure():
    cell = E.study_cell_run(1, 80, 1.0, 4.0, 10_000, 5, 3)
    assert cell["reps"] == 3 and len(cell["waits"]) == len(cell["wait_probs"]) == 3 and cell["exact"] == pytest.approx(10.0) and cell["n"] == 10_000


def test_precomputed_file_is_complete():
    pre = E.load_precomputed()
    assert {(x["c"], x["rho_pct"], x["ca2"], x["cs2"]) for x in pre["study"]} == {(c, r, a, s) for c in C.STUDY_C for r in C.STUDY_RHO_PCT for a in C.STUDY_CA2 for s in C.STUDY_CS2}
    assert {(x["c"], x["rho_pct"], x["cs2"]) for x in pre["effort"]} == {(c, r, s) for c in C.STUDY_C for r in C.STUDY_RHO_PCT for s in C.STUDY_CS2}
    assert pre["study_customers"] == C.STUDY_CUSTOMERS and pre["study_reps"] == C.STUDY_REPS
    assert pre["effort_customers"] == C.EFFORT_CUSTOMERS and pre["effort_reps"] == C.EFFORT_REPS
    for x in pre["study"]:
        assert x["reps"] == C.STUDY_REPS and len(x["waits"]) == C.STUDY_REPS and x["n"] == C.STUDY_CUSTOMERS
        assert x["approx"] == pytest.approx(F.approx_wait(x["c"], x["rho_pct"] / 100, x["ca2"], x["cs2"]))
    for x in pre["effort"]:
        assert x["reps"] == C.EFFORT_REPS and len(x["waits"]) == C.EFFORT_REPS and x["n"] == C.EFFORT_CUSTOMERS and x["ca2"] == 1.0


def test_lookups_and_missing_cells():
    pre = E.load_precomputed()
    assert E.study_cell(pre, 1, 80, 1.0, 4.0)["cs2"] == 4.0 and E.effort_cell(pre, 4, 95, 0.25)["c"] == 4
    with pytest.raises(KeyError):
        E.study_cell(pre, 2, 80, 1.0, 4.0)
    with pytest.raises(KeyError):
        E.effort_cell(pre, 1, 60, 1.0)
