"""Auswertung: Live-Lauf gegen Formeln, Studienzellen (Wiederholungen: Mittel, Streuung, Fehler der Näherungen, Aufwand für eine Genauigkeit). Die teure Studie
steht vorgerechnet in `precomputed_sweep.json` (Generator: generate_precomputed.py, Laden: `load_precomputed`)."""

import json
import math
from pathlib import Path

import mg1_constants as C
import mg1_formulas as F
from mg1_simulation import simulate

PRECOMPUTED_PATH = Path(__file__).resolve().parent / "precomputed_sweep.json"


def mean_and_sd(values):
    """Mittelwert und Stichproben-Standardabweichung (0.0 bei nur einem Wert)."""
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, 0.0
    return m, math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def formulas(c, rho_pct, ca2, cs2):
    """Alle Formelwerte einer Einstellung in Abfertigungsdauern: Näherung (Kingman bzw. Allen-Cunneen), exakter Wert (oder None), M/M/c als Bezug."""
    rho = rho_pct / 100.0
    return {"approx": F.approx_wait(c, rho, ca2, cs2), "exact": F.exact_wait(c, rho, ca2, cs2), "mm": F.mmc_wait(c, rho)}


def live_report(c, rho_pct, ca2, cs2, seed, customers=C.LIVE_CUSTOMERS):
    """Ein Live-Lauf mit den Formelwerten."""
    sim = simulate(c, rho_pct / 100.0, ca2, cs2, customers, seed)
    out = formulas(c, rho_pct, ca2, cs2)
    out.update({"sim": sim, "wait_sim": sim.mean_wait})
    return out


def study_cell_run(c, rho_pct, ca2, cs2, n, seed, reps):
    """Eine Studienzelle: `reps` unabhängige Läufe (Seeds seed, seed + 1000, …) mit den Formelwerten."""
    runs = [simulate(c, rho_pct / 100.0, ca2, cs2, n, seed + 1000 * r) for r in range(reps)]
    return summarize_runs(c, rho_pct, ca2, cs2, n, runs)


def summarize_runs(c, rho_pct, ca2, cs2, n, runs):
    out = {"c": c, "rho_pct": rho_pct, "ca2": ca2, "cs2": cs2, "n": n, "reps": len(runs), "waits": [r.mean_wait for r in runs],
           "wait_probs": [r.wait_prob for r in runs]}
    out.update(formulas(c, rho_pct, ca2, cs2))
    return out


def cell_mean(cell):
    return sum(cell["waits"]) / len(cell["waits"])


def cell_se(cell):
    """Standardfehler des Mittels der Wiederholungen."""
    return mean_and_sd(cell["waits"])[1] / math.sqrt(len(cell["waits"]))


def cell_rel_sd(cell):
    """Relative Standardabweichung eines Laufs (sd/Mittel); nan bei Mittel null (z. B. fest/fest)."""
    m, sd = mean_and_sd(cell["waits"])
    return sd / m if m > 1e-12 else float("nan")


def approx_error(cell):
    """Fehler der Näherung gegen die Simulation: Näherung/Simulation − 1; None, wenn die Simulation null wartet."""
    return F.relative_error(cell["approx"], cell_mean(cell))


def exact_error(cell):
    """Abweichung der Simulation vom exakten Wert (Simulation/exakt − 1); None, wo es keine Formel gibt."""
    return None if cell["exact"] is None else F.relative_error(cell_mean(cell), cell["exact"])


def customers_needed(cell, target=C.TARGET_REL_ERR):
    """Kunden für die relative Standardabweichung `target` eines Laufs, hochgerechnet aus der Streuung der Wiederholungen."""
    sd = cell_rel_sd(cell)
    return F.customers_for_precision(sd, cell["n"], target) if sd == sd else float("nan")


def load_precomputed():
    return json.loads(PRECOMPUTED_PATH.read_text(encoding="utf-8"))


def study_cell(pre, c, rho_pct, ca2, cs2):
    for cell in pre["study"]:
        if cell["c"] == c and cell["rho_pct"] == rho_pct and cell["ca2"] == ca2 and cell["cs2"] == cs2:
            return cell
    raise KeyError((c, rho_pct, ca2, cs2))


def effort_cell(pre, c, rho_pct, cs2):
    """Zelle der Aufwandsmessung (Poisson-Ankünfte, 48 Läufe à 100 000 Kunden)."""
    for cell in pre["effort"]:
        if cell["c"] == c and cell["rho_pct"] == rho_pct and cell["cs2"] == cs2:
            return cell
    raise KeyError((c, rho_pct, cs2))


def nearest(options, value):
    """Nächster Wert aus `options` (bei Gleichstand der kleinere)."""
    return min(options, key=lambda o: (abs(o - value), o))
