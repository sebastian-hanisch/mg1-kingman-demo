"""Presets: Vollständigkeit, gültige Werte, Permalink-Konstanten."""

import pytest

import mg1_constants as C
import mg1_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_ORDER) and len(C.PRESETS) == 4
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for preset in C.PRESETS.values():
        assert preset["c"] in C.C_OPTIONS and preset["ca2"] in C.CA2_OPTIONS and preset["cs2"] in C.CS2_OPTIONS
        assert C.RHO_PCT_MIN <= preset["rho_pct"] <= C.RHO_PCT_MAX and P.snap_rho(preset["rho_pct"]) == preset["rho_pct"]
        for key, state_key in P.PRESET_KEYS.items():
            P.SETTING_SPECS[state_key].caster(preset[key])


def test_preset_names_state_the_values_they_set():
    assert C.PRESETS["Feste Dauer (M/D/1)"]["cs2"] == 0.0 and C.PRESETS["Feste Dauer (M/D/1)"]["ca2"] == 1.0
    assert C.PRESETS["Streuende Dauer (cs² = 4)"]["cs2"] == 4.0 and C.PRESETS["Vier Spuren"]["c"] == 4
    t = C.PRESETS["Termine (glatte Ankünfte)"]
    assert (t["ca2"], t["cs2"], t["rho_pct"]) == (0.0, 1.0, 50)


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Streuende Dauer (cs² = 4)"]
    assert (p["c"], p["rho_pct"], p["ca2"], p["cs2"], p["seed"]) == (C.DEFAULT_C, C.DEFAULT_RHO_PCT, C.DEFAULT_CA2, C.DEFAULT_CS2, C.DEFAULT_SEED)


def test_every_preset_lies_on_a_study_cell():
    """Die Hilfetexte nennen Zahlen der Studie, also müssen Presets auf Studienzellen liegen."""
    for p in C.PRESETS.values():
        assert p["c"] in C.STUDY_C and p["rho_pct"] in C.STUDY_RHO_PCT and p["ca2"] in C.STUDY_CA2 and p["cs2"] in C.STUDY_CS2


def test_bounds_and_url_params():
    assert P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("rho_slider") == (50, 95)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


@pytest.mark.parametrize("key,value,expected", [("c_select", 2, 1), ("c_select", 3, 4), ("c_select", 9, 4), ("ca2_select", 0.4, 0.0), ("ca2_select", 0.6, 1.0),
                                                ("ca2_select", 2.0, 1.0), ("ca2_select", 3.0, 4.0), ("cs2_select", 0.1, 0.0), ("cs2_select", 0.5, 0.25),
                                                ("cs2_select", 2.0, 1.0), ("cs2_select", 2.6, 4.0)])
def test_option_regulators_snap_to_the_nearest_option(key, value, expected):
    assert P.snap_to_option(key, value) == expected


@pytest.mark.parametrize("value,expected", [(0, 50), (52, 50), (53, 55), (87, 85), (88, 90), (99, 95)])
def test_utilisation_snaps_to_the_step_inside_the_bounds(value, expected):
    assert P.snap_rho(value) == expected


def test_formatters():
    assert C.fmt_int(150000) == "150.000" and C.fmt_pct(0.2) == "20 %" and C.fmt_pct(0.0898, 1) == "9.0 %"
    assert C.fmt_signed_pct(0.17) == "+17 %" and C.fmt_signed_pct(-0.052, 1) == "−5.2 %" and C.fmt_signed_pct(0.0) == "+0 %"
