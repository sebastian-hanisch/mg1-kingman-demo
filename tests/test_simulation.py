"""Simulation: Sampler einzeln (von Hand und Verteilungsgrößen), Mini-Instanzen von Hand (eine und zwei Spuren), Lindley-Rekursion als exakte Gegenprobe, Vergleich gegen die
exakten Formeln (M/G/1, G/M/1, M/M/c), Invarianten und Reproduzierbarkeit."""

import pytest

import mg1_constants as C
import mg1_formulas as F
import mg1_simulation as S
from conftest import ScriptedRng


# ---------------------------------------------------------------- Einheit: Sampler

def test_sampler_by_hand():
    assert S.make_sampler(2.5, 0.0, ScriptedRng())() == 2.5                                              # fest
    assert S.make_sampler(2.0, 1.0, ScriptedRng(exp_values=[0.7]))() == 0.7                              # exponentiell: der Skriptwert
    assert S.make_sampler(1.0, 0.25, ScriptedRng(exp_values=[0.1, 0.2, 0.3, 0.4]))() == pytest.approx(1.0)   # Erlang-4: Summe von vier Phasen
    h2 = S.make_sampler(1.0, 4.0, ScriptedRng(exp_values=[0.6, 0.9], uniform_values=[0.5, 0.95]))
    assert (h2(), h2()) == (0.6, 0.9)                                                                    # u = 0.5 < p₁ → Phase 1, u = 0.95 > p₁ → Phase 2


def test_balanced_h2_probability_by_hand():
    assert S.balanced_h2_probability(1.0) == pytest.approx(0.5) and S.balanced_h2_probability(4.0) == pytest.approx(0.5 * (1 + (3 / 5) ** 0.5))


@pytest.mark.parametrize("scv", C.CS2_OPTIONS)
def test_every_sampler_has_mean_one_and_the_requested_scv(scv):
    rng = S.SplitMix64(7)
    draw = S.make_sampler(1.0, scv, rng)
    xs = [draw() for _ in range(600_000)]
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    assert mean == pytest.approx(1.0, abs=0.01)
    assert var / mean ** 2 == pytest.approx(scv, abs=0.06 if scv > 1 else 0.02)


def test_sampler_scales_with_the_mean():
    rng = S.SplitMix64(3)
    xs = [S.make_sampler(2.5, 4.0, rng)() for _ in range(300_000)]
    assert sum(xs) / len(xs) == pytest.approx(2.5, rel=0.02)


# ---------------------------------------------------------------- Mini-Instanzen von Hand

def _scripted(fixture):
    gaps, svcs = fixture
    return (lambda: gaps.expovariate(1.0)), (lambda: svcs.expovariate(1.0))


def test_one_lane_by_hand(two_customers_one_lane):
    """Siehe conftest: Wartezeiten 0 / 1.5 / 2.0 / 0, Mittel 0.875, die Hälfte der Kunden wartet; Summe der Dauern 3.7, Zeitraum von der ersten bis zur letzten Ankunft 4.0."""
    res = S.simulate(1, 0.5, 1.0, 1.0, 4, seed=0, warm_fraction=0.0, samplers=_scripted(two_customers_one_lane))
    assert res.n == 4 and res.mean_wait == pytest.approx(0.875) and res.wait_prob == pytest.approx(0.5)
    assert res.utilisation == pytest.approx(3.7 / 4.0) and res.mean_interarrival == pytest.approx(5.0 / 4) and res.mean_service == pytest.approx(3.7 / 4)


def test_two_lanes_by_hand():
    """c = 2, Ankünfte bei 0.5 / 1.0 / 1.5, Dauern 2.0 / 2.0 / 1.0: Kunde 1 startet bei 0.5 (frei 2.5), Kunde 2 bei 1.0 (frei 3.0), Kunde 3 wartet bis 2.5 (Wartezeit 1.0)."""
    gaps, svcs = ScriptedRng(exp_values=[0.5, 0.5, 0.5]), ScriptedRng(exp_values=[2.0, 2.0, 1.0])
    res = S.simulate(2, 0.5, 1.0, 1.0, 3, seed=0, warm_fraction=0.0, samplers=((lambda: gaps.expovariate(1.0)), (lambda: svcs.expovariate(1.0))))
    assert res.mean_wait == pytest.approx(1 / 3) and res.wait_prob == pytest.approx(1 / 3)


def test_warm_up_customers_are_not_evaluated():
    gaps, svcs = ScriptedRng(exp_values=[1.0, 0.5, 0.5, 3.0]), ScriptedRng(exp_values=[2.0, 1.0, 0.5, 0.2])
    res = S.simulate(1, 0.5, 1.0, 1.0, 4, seed=0, warm_fraction=0.5, samplers=((lambda: gaps.expovariate(1.0)), (lambda: svcs.expovariate(1.0))))
    assert res.n == 2 and res.mean_wait == pytest.approx((2.0 + 0.0) / 2)                  # Kunden 3 und 4


# ---------------------------------------------------------------- Gegenprobe: Lindley-Rekursion

@pytest.mark.parametrize("rho,ca2,cs2", [(0.8, 1.0, 4.0), (0.5, 0.0, 1.0), (0.9, 4.0, 0.25), (0.7, 0.0, 0.0), (0.6, 4.0, 4.0)])
def test_the_lindley_recursion_gives_the_same_path_for_one_lane(rho, ca2, cs2):
    """Eine Spur: Kiefer-Wolfowitz (Heap) und Lindley (W' = max(0, W + S − A)) aus denselben Strömen liefern dieselbe mittlere Wartezeit (bis auf Rundung)."""
    n = 60_000
    assert S.simulate(1, rho, ca2, cs2, n, 11).mean_wait == pytest.approx(S.lindley_mean_wait(rho, ca2, cs2, n, 11), rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------- Invarianten

@pytest.mark.parametrize("c", [1, 4])
def test_utilisation_and_observed_means_match_the_inputs(c):
    r = S.simulate(c, 0.8, 1.0, 1.0, 200_000, 5)
    assert r.utilisation == pytest.approx(0.8, abs=0.015) and r.mean_service == pytest.approx(1.0, abs=0.01)
    assert r.mean_interarrival * c * 0.8 == pytest.approx(1.0, abs=0.012)


def test_same_seed_same_result_and_different_seed_differs():
    a, b, c = S.simulate(4, 0.8, 1.0, 4.0, 20_000, 5), S.simulate(4, 0.8, 1.0, 4.0, 20_000, 5), S.simulate(4, 0.8, 1.0, 4.0, 20_000, 6)
    assert a == b and a.mean_wait != c.mean_wait


def test_no_waiting_without_any_variability_below_full_load():
    """Feste Ankünfte und feste Dauer bei ρ < 1 (D/D/1): nie eine Wartezeit."""
    r = S.simulate(1, 0.8, 0.0, 0.0, 5_000, 1)
    assert r.mean_wait == 0.0 and r.wait_prob == 0.0


# ---------------------------------------------------------------- Referenz: exakte Formeln

def _band(c, rho, ca2, cs2, reps=4, n=300_000, seed0=100):
    runs = [S.simulate(c, rho, ca2, cs2, n, seed0 + 17 * r).mean_wait for r in range(reps)]
    mean = sum(runs) / reps
    se = (sum((x - mean) ** 2 for x in runs) / (reps - 1) / reps) ** 0.5
    return mean, se


@pytest.mark.parametrize("cs2", C.CS2_OPTIONS)
def test_simulation_matches_pollaczek_khinchine(cs2):
    exact = F.pk_wait(0.8, cs2)
    mean, se = _band(1, 0.8, 1.0, cs2)
    assert abs(mean - exact) < max(4 * se, 0.04 * exact), (cs2, exact, mean, se)


@pytest.mark.parametrize("ca2", [0.0, 0.25, 4.0])
def test_simulation_matches_the_exact_gim1_wait(ca2):
    exact = F.gim1_wait(0.8, ca2)
    mean, se = _band(1, 0.8, ca2, 1.0)
    assert abs(mean - exact) < max(4 * se, 0.04 * exact), (ca2, exact, mean, se)


def test_simulation_matches_erlang_c_for_four_lanes():
    exact = F.mmc_wait(4, 0.8)
    mean, se = _band(4, 0.8, 1.0, 1.0, n=400_000)
    assert abs(mean - exact) < max(4 * se, 0.04 * exact), (exact, mean, se)
    pw = S.simulate(4, 0.8, 1.0, 1.0, 400_000, 3).wait_prob
    assert pw == pytest.approx(F.erlang_c(4, 3.2), abs=0.01)                              # P(Warten) = Erlang C


def test_variability_of_the_service_time_raises_the_wait_in_the_simulation():
    waits = [S.simulate(1, 0.8, 1.0, cs2, 200_000, 9).mean_wait for cs2 in C.CS2_OPTIONS]
    assert all(a < b for a, b in zip(waits, waits[1:]))
