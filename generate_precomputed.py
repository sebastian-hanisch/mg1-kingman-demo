"""Rechnet die teure Studie vor (Build-Zeit, nicht in der App): `python generate_precomputed.py [Prozesse]` schreibt
`precomputed_sweep.json`.

  study   Spuren (1, 4) × Auslastung (50, 80, 95 %) × Streuung der Ankünfte (0, 1, 4) × Streuung der Dauer (0, 0.25, 1, 4): je 6 Läufe à 1 500 000 Kunden;
          mittlere Wartezeit, Anteil Wartender, dazu die Formelwerte (Näherung, exakter Wert, M/M/c)
  effort  Poisson-Ankünfte, Spuren × Auslastung × Streuung der Dauer: je 48 Läufe à 100 000 Kunden (Streuung eines Lauf für die Aufwandsrechnung)"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mg1_constants as C
from mg1_evaluation import PRECOMPUTED_PATH, summarize_runs
from mg1_simulation import simulate


def _task(args):
    key, c, rho_pct, ca2, cs2, seed = args
    n = C.EFFORT_CUSTOMERS if key[0] == "effort" else C.STUDY_CUSTOMERS
    return key, simulate(c, rho_pct / 100.0, ca2, cs2, n, seed)


def main(workers):
    t0 = time.time()
    jobs, idx = [], 0
    for c in C.STUDY_C:
        for rho_pct in C.STUDY_RHO_PCT:
            for ca2 in C.STUDY_CA2:
                for cs2 in C.STUDY_CS2:
                    for r in range(C.STUDY_REPS):
                        jobs.append((("study", c, rho_pct, ca2, cs2), c, rho_pct, ca2, cs2, 10_000 + 97 * idx + 1000 * r))
                    idx += 1
    for c in C.STUDY_C:
        for rho_pct in C.STUDY_RHO_PCT:
            for cs2 in C.STUDY_CS2:
                for r in range(C.EFFORT_REPS):
                    jobs.append((("effort", c, rho_pct, 1.0, cs2), c, rho_pct, 1.0, cs2, 400_000 + 31 * idx + 13 * r))
                idx += 1
    jobs.sort(key=lambda j: -j[1])                       # Mehrspur-Läufe (langsamer) zuerst
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_task, jobs, chunksize=1))
    groups = {}
    for key, res in out:
        groups.setdefault(key, []).append(res)
    study, effort = [], []
    for (which, c, rho_pct, ca2, cs2), runs in groups.items():
        n = C.EFFORT_CUSTOMERS if which == "effort" else C.STUDY_CUSTOMERS
        (effort if which == "effort" else study).append(summarize_runs(c, rho_pct, ca2, cs2, n, runs))
    study.sort(key=lambda x: (x["c"], x["rho_pct"], x["ca2"], x["cs2"]))
    effort.sort(key=lambda x: (x["c"], x["rho_pct"], x["cs2"]))
    Path(PRECOMPUTED_PATH).write_text(json.dumps({"study_customers": C.STUDY_CUSTOMERS, "study_reps": C.STUDY_REPS, "effort_customers": C.EFFORT_CUSTOMERS,
                                                  "effort_reps": C.EFFORT_REPS, "study": study, "effort": effort}), encoding="utf-8")
    print(f"fertig in {time.time() - t0:.0f} s -> {PRECOMPUTED_PATH}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else min(6, os.cpu_count() or 1))
