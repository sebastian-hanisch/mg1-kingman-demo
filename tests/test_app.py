"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Randwerte, Würfel-Knopf, Permalink-Grenzen, Abschnitte, Footer."""

import random
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import mg1_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_default_run_has_no_exception_and_shows_the_exact_values():
    at = _run()
    _ok(at)
    assert _metric(at, "Exakter Wert") == "30.00 min" and _metric(at, "Näherung (Kingman bzw. Allen-Cunneen)") == "30.00 min"     # Pollaczek-Khinchine, ρ = 0.8, cs² = 4
    assert _metric(at, "M/M/1 (Bezug)") == "12.00 min"
    assert float(_metric(at, "Wartezeit (Simulation)").split()[0]) > 0 and _metric(at, "Simulation gegenüber M/M").endswith("-fach")


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert (at.session_state["c_select"], at.session_state["rho_slider"], at.session_state["ca2_select"], at.session_state["cs2_select"]) == (
        p["c"], p["rho_pct"], p["ca2"], p["cs2"])
    assert at.metric


def test_exact_value_is_shown_only_where_a_formula_exists():
    assert _metric(_run(c_select=4, ca2_select=0.0, cs2_select=4.0), "Exakter Wert") == "keine Formel"
    assert _metric(_run(c_select=1, ca2_select=0.0, cs2_select=1.0, rho_slider=50), "Exakter Wert") == "0.77 min"              # D/M/1 (G/M/1 exakt)
    assert _metric(_run(c_select=4, ca2_select=1.0, cs2_select=1.0), "Exakter Wert").endswith("min")                          # M/M/c (Erlang C)


def test_more_variability_of_the_service_time_lengthens_the_approximate_wait():
    wait = lambda at: float(_metric(at, "Näherung (Kingman bzw. Allen-Cunneen)").split()[0])
    waits = [wait(_run(cs2_select=cs2)) for cs2 in C.CS2_OPTIONS]
    assert all(a < b for a, b in zip(waits, waits[1:]))


@pytest.mark.parametrize("kw", [dict(c_select=1, rho_slider=95, ca2_select=4.0, cs2_select=4.0), dict(c_select=4, rho_slider=50, ca2_select=0.0, cs2_select=0.0),
                                 dict(c_select=4, rho_slider=95, ca2_select=4.0, cs2_select=0.25), dict(c_select=1, rho_slider=50, ca2_select=0.0, cs2_select=0.0),
                                 dict(c_select=1, rho_slider=65, ca2_select=1.0, cs2_select=0.25)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_dice_button_changes_the_seed_and_the_simulated_run(monkeypatch):
    """Der Würfel zieht sonst einen unseeded Zufalls-Seed; deshalb ist der gewürfelte Seed im Test fest. Verglichen wird die Simulation selbst (Metrik mit zwei Nachkommastellen)."""
    monkeypatch.setattr(random, "randint", lambda a, b: 508145)
    at = _run()
    old_seed, old = at.session_state["seed_input"], _metric(at, "Wartezeit (Simulation)")
    next(b for b in at.button if b.label == "🎲 Neuen Lauf würfeln").click().run()
    _ok(at)
    assert at.session_state["seed_input"] == 508145 != old_seed and _metric(at, "Wartezeit (Simulation)") != old


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["rho"] = "87"
    at.query_params["c"] = "3"
    at.query_params["ca2"] = "0.4"
    at.query_params["cs2"] = "0.5"
    at.run()
    _ok(at)
    assert at.session_state["rho_slider"] == 85 and at.session_state["c_select"] == 4 and at.session_state["ca2_select"] == 0.0
    assert at.session_state["cs2_select"] == 0.25


def test_permalink_ignores_garbage():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["rho"] = "viel"
    at.query_params["cs2"] = "nan"
    at.query_params["c"] = "x"
    at.run()
    _ok(at)
    assert at.session_state["rho_slider"] == C.DEFAULT_RHO_PCT and at.session_state["cs2_select"] == C.DEFAULT_CS2 and at.session_state["c_select"] == C.DEFAULT_C


def test_charts_sections_and_limits_table_are_present():
    at = _run()
    _ok(at)
    assert len(at.get("plotly_chart")) == 5
    headers = [s.value for s in at.subheader]
    for part in ("Wie genau sind Kingman", "Was kostet Genauigkeit", "Wo die Annahmen enden"):
        assert any(part in h for h in headers), part
    table = next(m.value for m in at.markdown if "Wer setzt an" in m.value)
    for name in ("Zeitvariable Ankünfte", "Prioritätsklassen", "Jackson-Netze", "Erlang A", "Erlang B"):
        assert name in table
    assert "geplant" not in table      # die Linie wird erst vollständig veröffentlicht, kein Status-Zusatz


def test_error_table_is_complete():
    at = _run()
    table = next(m.value for m in at.markdown if m.value.startswith("| Ankünfte | Dauer | Simulation"))
    assert table.count("\n| ") == len(C.STUDY_CA2) * len(C.STUDY_CS2)
    for a in ("glatt", "Poisson", "stoßweise"):
        assert f"| {a} |" in table


def test_seed_control_uses_the_portfolio_wording():
    at = _run()
    assert [n.label for n in at.number_input] == ["Zufalls-Seed"]


def test_related_demos_are_linked_and_footer_is_present():
    at = _run()
    text = " ".join(c.value for c in at.caption)
    for name in ("mm1-queue-demo", "mmc-queue-demo", "erlang-b-demo", "output-analysis-demo", "truck-appointment-demo"):
        assert name in text
    assert "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in text


def test_no_sentence_wide_comma_replacement_in_the_app_source():
    """Regressionsschutz: `.replace(",", ".")` auf einem ganzen (verketteten) Satz macht aus Kommas im Fließtext Punkte; Tausender nur über `fmt_int`."""
    source = Path(APP).read_text(encoding="utf-8")
    assert '.replace(",", ".")' not in source
