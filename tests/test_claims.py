"""JEDE Zahl aus README und App-Texten wird hier nachgerechnet: Formelwerte exakt, Studien-Zahlen aus der vorgerechneten Datei (je 6 Läufe à 1 500 000 Kunden,
Aufwandsmessung je 48 Läufe à 100 000 Kunden). Die Datei ist fest; ändern sich die Zahlen nach einer neuen Rechnung, müssen README und Hilfetexte nachgezogen werden.
Wartezeiten in Abfertigungsdauern, wo nicht Minuten (3 min Mittel) dasteht."""

import pytest

import mg1_constants as C
import mg1_evaluation as E
import mg1_formulas as F

PRE = E.load_precomputed()


def cell(c, rho, ca2, cs2):
    return E.study_cell(PRE, c, rho, ca2, cs2)


def pct(x, digits=0):
    return round(100 * x) if digits == 0 else round(100 * x, digits)


def test_pollaczek_khinchine_against_simulation_at_load_80():
    """README (eine Spur, Poisson-Ankünfte, Auslastung 80 %): Wartezeit nach Pollaczek-Khinchine 2.0 / 2.5 / 4.0 / 10.0 für cs² = 0 / 0.25 / 1 / 4; Simulation 2.004 / 2.500 /
    3.987 / 9.878; das 0.50- / 0.62- / 1.00- / 2.47-Fache von M/M/1."""
    assert [F.pk_wait(0.8, s) for s in C.CS2_OPTIONS] == pytest.approx([2.0, 2.5, 4.0, 10.0])
    sims = [E.cell_mean(cell(1, 80, 1.0, s)) for s in C.CS2_OPTIONS]
    assert [round(x, 3) for x in sims] == pytest.approx([2.004, 2.500, 3.987, 9.878], abs=0.0005)
    assert [round(x / F.mm1_wait(0.8), 2) for x in sims] == [0.50, 0.62, 1.00, 2.47]


def test_simulation_agrees_with_every_exact_formula():
    """README: In allen 21 Zellen, für die es eine exakte Formel gibt (Pollaczek-Khinchine, G/M/1, Erlang C), weicht die Simulation höchstens 3.7 % ab, im Mittel 0.6 %."""
    errs = [abs(E.exact_error(x)) for x in PRE["study"] if E.exact_error(x) is not None]
    assert len(errs) == 21 and pct(max(errs), 1) == 3.7 and pct(sum(errs) / len(errs), 1) == 0.6


def test_kingman_is_exact_for_poisson_arrivals_and_one_lane():
    """README: Bei einer Spur und Poisson-Ankünften ist Kingman gleich Pollaczek-Khinchine; die Simulation weicht in allen zwölf Zellen höchstens 3.9 % ab."""
    errs = [abs(E.approx_error(cell(1, r, 1.0, s))) for r in C.STUDY_RHO_PCT for s in C.STUDY_CS2]
    assert len(errs) == 12 and pct(max(errs), 1) == 3.9
    for r in C.STUDY_RHO_PCT:
        for s in C.STUDY_CS2:
            assert cell(1, r, 1.0, s)["approx"] == pytest.approx(F.pk_wait(r / 100, s))


def test_kingman_error_shrinks_with_the_load_for_smooth_and_bursty_arrivals():
    """README (eine Spur, Auslastung 50 / 80 / 95 %): Kingman gegen Simulation bei glatten Ankünften und exponentieller Dauer (D/M/1) +96 / +18 / +2.4 %, bei stoßweisen
    Ankünften und fester Dauer +52 / +7.6 / +0.7 %, bei stoßweisen Ankünften und exponentieller Dauer +16 / +1.6 / −0.1 %; der Fehler sinkt jeweils mit der Auslastung."""
    err = lambda a, s: [E.approx_error(cell(1, r, a, s)) for r in C.STUDY_RHO_PCT]
    assert [pct(x, 1) for x in err(0.0, 1.0)] == pytest.approx([96.1, 18.1, 2.4], abs=0.051)
    assert [pct(x, 1) for x in err(4.0, 0.0)] == pytest.approx([51.7, 7.6, 0.7], abs=0.051)
    assert [pct(x, 1) for x in err(4.0, 1.0)] == pytest.approx([15.8, 1.6, -0.1], abs=0.051)
    for a, s in ((0.0, 1.0), (4.0, 0.0), (4.0, 1.0)):
        e = [abs(x) for x in err(a, s)]
        assert e[0] > e[1] > e[2], (a, s)


def test_dm1_simulation_against_the_exact_gim1_wait():
    """README (D/M/1): Simulation 0.255 / 1.694 / 9.281 gegen exakt (G/M/1) 0.255 / 1.693 / 9.172 bei Auslastung 50 / 80 / 95 %."""
    cells = [cell(1, r, 0.0, 1.0) for r in C.STUDY_RHO_PCT]
    assert [round(E.cell_mean(x), 3) for x in cells] == pytest.approx([0.255, 1.694, 9.281], abs=0.0005)
    assert [round(x["exact"], 3) for x in cells] == pytest.approx([0.255, 1.693, 9.172], abs=0.0005)


def test_allen_cunneen_for_four_lanes_is_good_at_high_load_and_poor_at_low_load():
    """README (vier Spuren): relativ bis +2552 % bei Auslastung 50 % (D/M/4: 0.043 gegen 0.010), aber absolut höchstens 0.3 min; bei 80 % bis 77 %, absolut 0.9 min; bei 95 % alle
    Zellen mit Wartezeit innerhalb von ±11.1 % (absolut bis 2.1 min). M/M/4 (Erlang C) stimmt auf −0.1 / +0.3 / −1.7 %."""
    for r, rel, absolute in ((50, 2551.7, 0.3), (80, 77.0, 0.9), (95, 11.1, 2.1)):
        cells = [cell(4, r, a, s) for a in C.STUDY_CA2 for s in C.STUDY_CS2]
        errs = [abs(E.approx_error(x)) for x in cells if E.approx_error(x) is not None]
        assert pct(max(errs), 1) == pytest.approx(rel, abs=0.06 * (rel / 100 if rel > 1000 else 1))
        assert round(F.to_minutes(max(abs(x["approx"] - E.cell_mean(x)) for x in cells)), 1) == absolute
    assert [pct(E.approx_error(cell(4, r, 1.0, 1.0)), 1) for r in C.STUDY_RHO_PCT] == pytest.approx([-0.1, 0.3, -1.7], abs=0.051)
    assert round(cell(4, 50, 0.0, 1.0)["approx"], 3) == 0.043 and round(E.cell_mean(cell(4, 50, 0.0, 1.0)), 3) == 0.010


def test_the_smoothing_arrivals_are_overestimated_and_bursty_arrivals_underestimated_for_several_lanes():
    """README (vier Spuren, Auslastung 50 %): glatte Ankünfte +323 % (exponentielle Dauer), stoßweise Ankünfte −26 %, stoßweise Dauer bei Poisson-Ankünften +30 %."""
    assert (pct(E.approx_error(cell(4, 50, 0.0, 1.0))), pct(E.approx_error(cell(4, 50, 4.0, 1.0))), pct(E.approx_error(cell(4, 50, 1.0, 4.0)))) == (323, -26, 30)


def test_effort_for_five_percent_precision():
    """README (eine Spur, Poisson-Ankünfte, je 48 Läufe à 100 000 Kunden): Kunden für ±5 % bei Auslastung 50 / 80 / 95 % und cs² = 0 / 0.25 / 1 / 4 in Tausend gerundet:
    8 / 9 / 15 / 41 ; 18 / 30 / 41 / 145 ; 305 / 361 / 800 / 1546. Je Reihe steigt der Aufwand mit cs² und mit der Auslastung."""
    need = {(r, s): E.customers_needed(E.effort_cell(PRE, 1, r, s)) for r in C.STUDY_RHO_PCT for s in C.STUDY_CS2}
    got = [[round(need[(r, s)] / 1000) for s in C.STUDY_CS2] for r in C.STUDY_RHO_PCT]
    assert got == [[8, 9, 15, 41], [18, 30, 41, 145], [305, 361, 800, 1546]]
    for r in C.STUDY_RHO_PCT:
        row = [need[(r, s)] for s in C.STUDY_CS2]
        assert all(a < b for a, b in zip(row, row[1:])), r
    for s in C.STUDY_CS2:
        col = [need[(r, s)] for r in C.STUDY_RHO_PCT]
        assert col[0] < col[1] < col[2], s
    assert round(need[(80, 4.0)] / need[(80, 0.0)], 1) == 8.2 and round(need[(95, 4.0)] / need[(95, 0.0)], 1) == 5.1


def test_noise_level_of_the_study_quoted_in_readme():
    """README: Die relative Streuung eines 1.5-Mio-Laufs beträgt höchstens 8.2 % (Mittel aus 6 Läufen: 3.3 %), meist unter 3 %."""
    sds = [E.cell_rel_sd(x) for x in PRE["study"] if E.cell_rel_sd(x) == E.cell_rel_sd(x)]
    assert pct(max(sds), 1) == 8.2 and round(100 * max(sds) / 6 ** 0.5, 1) == 3.3
    assert sum(1 for x in sds if x < 0.03) >= 0.7 * len(sds)


def test_preset_help_numbers():
    """PRESET_HELP: jede Zahl aus den vier Texten (siehe mg1_constants), Minuten bei 3 min Mittel."""
    h = C.PRESET_HELP
    m = lambda x, d=1: format(F.to_minutes(x), f".{d}f")
    t = h["Feste Dauer (M/D/1)"]
    x = cell(1, 80, 1.0, 0.0)
    assert m(E.cell_mean(x)) in t and m(F.mm1_wait(0.8)) in t and "12.0" in t and m(x["exact"]) == "6.0"
    t = h["Streuende Dauer (cs² = 4)"]
    x = cell(1, 80, 1.0, 4.0)
    assert m(E.cell_mean(x)) in t and m(x["exact"]) in t and "12.0" in t and round(x["exact"] / F.mm1_wait(0.8), 1) == 2.5
    t = h["Termine (glatte Ankünfte)"]
    x = cell(1, 50, 0.0, 1.0)
    assert m(E.cell_mean(x), 2) in t and m(x["exact"], 2) in t and m(x["approx"], 2) in t and f"{pct(E.approx_error(x))} %" in t
    t = h["Vier Spuren"]
    x = cell(4, 80, 1.0, 4.0)
    assert m(E.cell_mean(x)) in t and m(x["approx"]) in t and m(x["mm"]) in t and f"+{pct(E.approx_error(x))} %" in t


def test_default_live_run_numbers():
    """README: Standardlauf (eine Spur, 80 %, Poisson, cs² = 4, 150 000 Kunden, Seed 35): 32.01 min (exakt 30.00 min, +6.7 %), 2.67-faches von M/M/1, 81.2 % der Kunden warten."""
    r = E.live_report(1, 80, 1.0, 4.0, 35)
    assert round(F.to_minutes(r["wait_sim"]), 2) == 32.01 and round(F.to_minutes(r["exact"]), 2) == 30.0
    assert round(100 * (r["wait_sim"] / r["exact"] - 1), 1) == 6.7 and round(r["wait_sim"] / r["mm"], 2) == 2.67 and round(100 * r["sim"].wait_prob, 1) == 81.2
