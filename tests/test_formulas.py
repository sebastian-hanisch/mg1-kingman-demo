"""Formeln: Pollaczek-Khinchine von Hand, Kingman und Allen-Cunneen, Laplace-Transformierte der Ankunftsprozesse (Mittel und Streuung), G/M/1 exakt, Auswahl des
exakten Werts, Hilfsfunktionen."""

import math

import pytest

import mg1_formulas as F


def test_pk_by_hand():
    """ρ = 0.5: fest 0.5, exponentiell 1.0; ρ = 0.8, cs² = 4: 10; die Formel über E[S²] = 1 + cs² gibt dasselbe: λ·E[S²]/(2(1 − ρ)) mit λ = ρ."""
    assert F.pk_wait(0.5, 0.0) == pytest.approx(0.5) and F.pk_wait(0.5, 1.0) == pytest.approx(1.0) and F.pk_wait(0.8, 4.0) == pytest.approx(10.0)
    for rho, cs2 in ((0.3, 0.25), (0.9, 4.0)):
        assert F.pk_wait(rho, cs2) == pytest.approx(rho * (1 + cs2) / (2 * (1 - rho)))
    assert F.pk_wait(0.8, 1.0) == pytest.approx(4.0)                                # M/M/1: ρ/(1 − ρ)


def test_fixed_service_halves_the_exponential_wait_and_variance_scales_linearly():
    for rho in (0.5, 0.8, 0.95):
        assert F.pk_wait(rho, 0.0) == pytest.approx(F.mm1_wait(rho) / 2)
        assert F.pk_wait(rho, 4.0) == pytest.approx(2.5 * F.mm1_wait(rho))           # (1 + 4)/2
        assert F.pk_wait(rho, 0.25) - F.pk_wait(rho, 0.0) == pytest.approx(0.25 * F.mm1_wait(rho) / 2)   # linear in cs²


def test_kingman_by_hand_and_symmetry():
    """Kingman: ρ/(1 − ρ)·(ca² + cs²)/2, symmetrisch in ca² und cs²; bei ca² = 1 gleich Pollaczek-Khinchine."""
    assert F.kingman_wait(0.8, 0.0, 1.0) == pytest.approx(2.0) and F.kingman_wait(0.8, 4.0, 4.0) == pytest.approx(16.0)
    assert F.kingman_wait(0.8, 0.0, 0.0) == 0.0
    for ca2, cs2 in ((0.0, 4.0), (1.0, 0.25), (4.0, 0.0)):
        assert F.kingman_wait(0.9, ca2, cs2) == pytest.approx(F.kingman_wait(0.9, cs2, ca2))
    for cs2 in (0.0, 0.25, 1.0, 4.0):
        assert F.kingman_wait(0.7, 1.0, cs2) == pytest.approx(F.pk_wait(0.7, cs2))


def test_erlang_c_and_mmc_wait_by_hand():
    assert F.erlang_c(2, 1.0) == pytest.approx(1 / 3) and F.mmc_wait(2, 0.5) == pytest.approx(1 / 3)           # M/M/2, a = 1: Wq = C/(c(1 − ρ))
    assert F.mmc_wait(1, 0.8) == pytest.approx(F.mm1_wait(0.8))
    with pytest.raises(ValueError):
        F.erlang_c(3, 3.0)


def test_allen_cunneen_reduces_to_kingman_and_to_mmc():
    for rho, ca2, cs2 in ((0.5, 0.0, 1.0), (0.8, 4.0, 0.25), (0.95, 1.0, 4.0)):
        assert F.allen_cunneen_wait(1, rho, ca2, cs2) == pytest.approx(F.kingman_wait(rho, ca2, cs2))
    assert F.allen_cunneen_wait(4, 0.8, 1.0, 1.0) == pytest.approx(F.mmc_wait(4, 0.8))
    assert F.allen_cunneen_wait(4, 0.8, 0.0, 4.0) == pytest.approx(2.0 * F.mmc_wait(4, 0.8))
    assert F.approx_wait(4, 0.8, 0.0, 4.0) == F.allen_cunneen_wait(4, 0.8, 0.0, 4.0)


def test_more_lanes_wait_less_at_the_same_load():
    assert F.mmc_wait(1, 0.8) > F.mmc_wait(2, 0.8) > F.mmc_wait(4, 0.8) > F.mmc_wait(16, 0.8)


def test_balanced_hyperexponential_has_mean_one_and_the_requested_scv():
    """H2 mit gleichen Phasenanteilen am Mittel: Raten 2p₁ und 2p₂, Mittel p₁/(2p₁) + p₂/(2p₂) = 1, zweites Moment 1/(2p₁) + 1/(2p₂), scv = Moment − 1."""
    for scv in (1.0, 2.0, 4.0, 9.0):
        p1 = F.balanced_h2_probability(scv)
        p2 = 1 - p1
        assert p1 / (2 * p1) + p2 / (2 * p2) == pytest.approx(1.0)
        assert 1 / (2 * p1) + 1 / (2 * p2) - 1 == pytest.approx(scv)


@pytest.mark.parametrize("ca2", [0.0, 0.25, 1.0, 4.0])
def test_interarrival_transform_has_the_right_mean_and_scv(ca2):
    """A*(0) = 1; −A*'(0) = Mittel 1/ρ; A*''(0) = zweites Moment = (1 + scv)/ρ² (numerisch mit zentralen Differenzen)."""
    rho = 0.8
    lst = F.interarrival_lst(rho, ca2)
    h = 1e-4
    assert lst(0.0) == pytest.approx(1.0)
    mean = -(lst(h) - lst(-h)) / (2 * h)
    second = (lst(h) - 2 * lst(0.0) + lst(-h)) / h ** 2
    assert mean == pytest.approx(1 / rho, rel=1e-6) and second == pytest.approx((1 + ca2) / rho ** 2, rel=1e-3)


def test_gim1_reduces_to_mm1_and_solves_the_fixed_point():
    for rho in (0.3, 0.5, 0.9):
        assert F.gim1_sigma(rho, 1.0) == pytest.approx(rho, abs=1e-9) and F.gim1_wait(rho, 1.0) == pytest.approx(F.mm1_wait(rho), rel=1e-8)
    for rho in (0.5, 0.8, 0.95):                                                      # D/M/1: σ = exp(−(1 − σ)/ρ)
        s = F.gim1_sigma(rho, 0.0)
        assert 0 < s < 1 and s == pytest.approx(math.exp(-(1 - s) / rho), abs=1e-9)


def test_gim1_by_hand_for_deterministic_arrivals():
    """D/M/1 bei ρ = 0.5: σ ≈ 0.2032 (Lösung von σ = e^{−2(1−σ)}), Wq = σ/(1 − σ) ≈ 0.255; Kingman sagt 0.5 (+96 %)."""
    assert F.gim1_sigma(0.5, 0.0) == pytest.approx(0.2032, abs=2e-4) and F.gim1_wait(0.5, 0.0) == pytest.approx(0.2550, abs=5e-4)
    assert F.relative_error(F.kingman_wait(0.5, 0.0, 1.0), F.gim1_wait(0.5, 0.0)) == pytest.approx(0.96, abs=0.01)


def test_smoother_arrivals_wait_less_and_burstier_arrivals_wait_more_exactly():
    for rho in (0.5, 0.8, 0.95):
        waits = [F.gim1_wait(rho, ca2) for ca2 in (0.0, 0.25, 1.0, 4.0)]
        assert all(a < b for a, b in zip(waits, waits[1:]))


def test_kingman_error_vanishes_in_heavy_traffic_against_the_exact_gim1_wait():
    """Kingman überschätzt D/M/1 bei Last 50 % um 96 %, bei 80 % um 18 %, bei 95 % um 4 %; der Fehler sinkt mit der Last (G/M/1 exakt)."""
    errs = [F.relative_error(F.kingman_wait(r, 0.0, 1.0), F.gim1_wait(r, 0.0)) for r in (0.5, 0.8, 0.95, 0.99)]
    assert all(a > b > 0 for a, b in zip(errs, errs[1:])) and errs[-1] < 0.03
    assert [round(100 * e) for e in errs[:3]] == [96, 18, 4]


def test_exact_wait_selects_the_right_formula():
    assert F.exact_wait(1, 0.8, 1.0, 4.0) == pytest.approx(10.0)                      # M/G/1
    assert F.exact_wait(1, 0.5, 0.0, 1.0) == pytest.approx(F.gim1_wait(0.5, 0.0))     # G/M/1
    assert F.exact_wait(4, 0.8, 1.0, 1.0) == pytest.approx(F.mmc_wait(4, 0.8))        # M/M/c
    assert F.exact_wait(4, 0.8, 0.0, 4.0) is None and F.exact_wait(1, 0.8, 0.0, 4.0) is None and F.exact_wait(4, 0.8, 1.0, 4.0) is None


def test_relative_error_and_customers_for_precision():
    assert F.relative_error(1.1, 1.0) == pytest.approx(0.1) and F.relative_error(1.0, 0.0) is None
    assert F.customers_for_precision(0.10, 100_000, 0.05) == pytest.approx(400_000)
    assert F.customers_for_precision(0.05, 100_000, 0.05) == pytest.approx(100_000)
    assert F.to_minutes(2.0) == 6.0


def test_invalid_input_is_rejected():
    for bad in (0.0, 1.0, 1.2):
        for call in (lambda r: F.pk_wait(r, 1.0), lambda r: F.kingman_wait(r, 1.0, 1.0), lambda r: F.gim1_sigma(r, 0.0), lambda r: F.mmc_wait(2, r)):
            with pytest.raises(ValueError):
                call(bad)
