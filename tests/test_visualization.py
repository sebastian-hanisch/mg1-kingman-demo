"""Abbildungen: gesperrte Achsen, Zahl der Linien, Beschriftungen."""

import mg1_constants as C
import mg1_evaluation as E
import mg1_formulas as F
import mg1_visualization as V

PRE = E.load_precomputed()


def _locked(fig):
    return all(ax.fixedrange for ax in fig.select_xaxes()) and all(ax.fixedrange for ax in fig.select_yaxes())


def test_all_charts_lock_their_axes():
    rep = E.live_report(1, 80, 1.0, 4.0, seed=1, customers=5_000)
    figs = [V.build_pk_chart(80, 4.0, rep["wait_sim"]), V.build_compare_chart(rep, 1, 1.0, 4.0), V.build_error_heatmap(PRE, 1, 80),
            V.build_heavy_traffic_chart(PRE, 4, 1.0), V.build_effort_chart(PRE, 1)]
    assert all(_locked(f) for f in figs)


def test_short_labels():
    assert [V.ca2_short(x) for x in C.CA2_OPTIONS] == ["glatt", "Poisson", "stoßweise"]
    assert [V.cs2_short(x) for x in C.CS2_OPTIONS] == ["fest", "Erlang-4", "exponentiell", "streuend"]


def test_pk_chart_has_one_line_per_service_variability_plus_the_simulation_point():
    fig = V.build_pk_chart(80, 4.0, 9.9)
    assert len(fig.data) == len(C.CS2_OPTIONS) + 1 and len(V.build_pk_chart(80, 4.0, None).data) == len(C.CS2_OPTIONS)
    top = max(fig.data[3].y)
    assert top > max(fig.data[0].y)                       # cs² = 4 liegt über cs² = 0 (Pollaczek-Khinchine linear in 1 + cs²)


def test_compare_chart_has_an_exact_bar_only_where_a_formula_exists():
    rep_exact = E.live_report(1, 80, 1.0, 4.0, seed=1, customers=4_000)
    rep_none = E.live_report(4, 80, 0.0, 4.0, seed=1, customers=4_000)
    assert list(V.build_compare_chart(rep_exact, 1, 1.0, 4.0).data[0].x) == ["M/M/1 (Bezug)", "Näherung (Kingman)", "exakt", "Simulation"]
    assert list(V.build_compare_chart(rep_none, 4, 0.0, 4.0).data[0].x) == ["M/M/4 (Bezug)", "Näherung (Allen-Cunneen)", "Simulation"]
    assert V.build_compare_chart(rep_exact, 1, 1.0, 4.0).data[0].y[0] == F.to_minutes(F.mm1_wait(0.8))


def test_heatmap_has_one_cell_per_variability_pair_and_empty_cells_without_waiting():
    fig = V.build_error_heatmap(PRE, 1, 80)
    z = fig.data[0].z
    assert len(z) == len(C.STUDY_CA2) and all(len(row) == len(C.STUDY_CS2) for row in z)
    assert z[0][0] is None                                # fest/fest: nie eine Wartezeit, kein Fehler


def test_heavy_traffic_chart_has_one_line_per_arrival_variability():
    fig = V.build_heavy_traffic_chart(PRE, 1, 1.0)
    assert [t.name for t in fig.data] == [f"Ankünfte {V.ca2_short(a)}" for a in C.STUDY_CA2]
    assert all(abs(y) < 5 for y in fig.data[1].y)         # Poisson-Ankünfte, eine Spur, exponentielle Dauer: Näherung exakt


def test_effort_chart_has_one_line_per_load_and_more_customers_for_higher_load():
    fig = V.build_effort_chart(PRE, 1)
    assert len(fig.data) == len(C.STUDY_RHO_PCT)
    assert all(a < b < c for a, b, c in zip(fig.data[0].y, fig.data[1].y, fig.data[2].y))
