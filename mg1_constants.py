"""Konstanten der Demo zu M/G/1 und Kingman: Regler, Voreinstellungen, Studien-Achsen. Zeiteinheit der Rechnung ist die mittlere Abfertigungsdauer
(= 1); angezeigt werden Minuten bei 3 min Mittel."""


def fmt_int(n):
    """Ganzzahl mit Punkt als Tausendertrenner (10000 -> 10.000)."""
    return f"{n:,}".replace(",", ".")


def fmt_pct(x, digits=0):
    """Anteil als Prozent mit Leerzeichen (0.086 -> "9 %", mit digits=1 "8.6 %")."""
    return f"{x:.{digits}%}".replace("%", " %")


def fmt_signed_pct(x, digits=0):
    """Abweichung als vorzeichenbehaftetes Prozent (0.17 -> "+17 %", -0.05 -> "−5 %")."""
    return f"{x:+.{digits}%}".replace("%", " %").replace("-", "−")


MEAN_SERVICE_MIN = 3.0                                  # mittlere Abfertigungsdauer je Spur (Minuten)

RHO_PCT_MIN, RHO_PCT_MAX, RHO_PCT_STEP, DEFAULT_RHO_PCT = 50, 95, 5, 80       # Auslastung je Spur in Prozent
C_OPTIONS = (1, 4)                                      # Spuren
DEFAULT_C = 1
CA2_OPTIONS = (0.0, 1.0, 4.0)                           # Streuung der Zwischenankunftszeiten (Variationskoeffizient²): glatt / Poisson / stoßweise
DEFAULT_CA2 = 1.0
CS2_OPTIONS = (0.0, 0.25, 1.0, 4.0)                     # Streuung der Abfertigungsdauer: fest / Erlang-4 / exponentiell / Hyperexponential
DEFAULT_CS2 = 4.0
CA2_LABELS = {0.0: "glatt (fest, ca² = 0)", 1.0: "Poisson (ca² = 1)", 4.0: "stoßweise (ca² = 4)"}
CS2_LABELS = {0.0: "fest (cs² = 0)", 0.25: "Erlang-4 (cs² = 0.25)", 1.0: "exponentiell (cs² = 1)", 4.0: "streuend (cs² = 4)"}
SEED_MAX = 999999
DEFAULT_SEED = 35

LIVE_CUSTOMERS = 150_000                                # Kunden je Live-Lauf

# Vorgerechnete Studie (generate_precomputed.py)
STUDY_C = C_OPTIONS
STUDY_RHO_PCT = (50, 80, 95)
STUDY_CA2 = CA2_OPTIONS
STUDY_CS2 = CS2_OPTIONS
STUDY_CUSTOMERS = 1_500_000                             # Kunden je Wiederholung
STUDY_REPS = 6                                          # unabhängige Wiederholungen je Zelle
TARGET_REL_ERR = 0.05                                   # Genauigkeitsziel für die Aufwandsrechnung (±5 %)
EFFORT_CUSTOMERS = 100_000                              # Aufwandsmessung: Kunden je Lauf
EFFORT_REPS = 48                                        # Aufwandsmessung: Wiederholungen (die Streuung eines Laufs braucht viele)

PRESET_ORDER = ("Feste Dauer (M/D/1)", "Streuende Dauer (cs² = 4)", "Termine (glatte Ankünfte)", "Vier Spuren")


def _preset(c=DEFAULT_C, rho_pct=DEFAULT_RHO_PCT, ca2=DEFAULT_CA2, cs2=DEFAULT_CS2):
    return {"c": c, "rho_pct": rho_pct, "ca2": ca2, "cs2": cs2, "seed": DEFAULT_SEED}


PRESETS = {
    "Feste Dauer (M/D/1)": _preset(cs2=0.0),
    "Streuende Dauer (cs² = 4)": _preset(),
    "Termine (glatte Ankünfte)": _preset(rho_pct=50, ca2=0.0, cs2=1.0),
    "Vier Spuren": _preset(c=4, cs2=4.0),
}
# Zahlen aus der vorgerechneten Studie (je 6 Läufe à 1 500 000 Kunden, 3 min mittlere Abfertigung); tests/test_claims.py rechnet jede nach
PRESET_HELP = {
    "Feste Dauer (M/D/1)": "Auslastung 80 %, eine Spur, Poisson-Ankünfte: Wartezeit 6.0 min (Simulation) statt 12.0 min bei exponentieller Dauer, genau die Hälfte (Pollaczek-Khinchine).",
    "Streuende Dauer (cs² = 4)": "Auslastung 80 %, eine Spur, Poisson-Ankünfte: Wartezeit 29.6 min (Simulation, exakt 30.0 min) statt 12.0 min bei exponentieller Dauer, das 2.5-Fache.",
    "Termine (glatte Ankünfte)": "Auslastung 50 %, eine Spur, feste Zwischenankunftszeit, exponentielle Dauer: Wartezeit 0.77 min (exakt 0.77 min nach G/M/1); Kingman sagt 1.50 min, 96 % zu viel.",
    "Vier Spuren": "Auslastung 80 %, vier Spuren, Poisson-Ankünfte, cs² = 4: Wartezeit 5.3 min (Simulation); Allen-Cunneen sagt 5.6 min (+6 %); mit exponentieller Dauer wären es 2.2 min.",
}
