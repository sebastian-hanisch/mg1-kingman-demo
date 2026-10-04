"""Formeln zur Wartezeit bei beliebiger Streuung: ein Gate mit c Spuren, Ankünfte mit Variationskoeffizient² ca², Abfertigung mit cs², mittlere Dauer 1,
Auslastung ρ je Spur. Alle Wartezeiten in Abfertigungsdauern (mal 3 min).

**M/G/1** (Pollaczek-Khinchine, exakt): Wq = λ·E[S²]/(2(1 − ρ)) = ρ(1 + cs²)/(2(1 − ρ)) bei mittlerer Dauer 1. Bei fester Dauer (cs² = 0) die Hälfte von M/M/1.
**G/M/1** (exakt, Ankunftsprozess beliebig, Dauer exponentiell): Wq = σ/(1 − σ), σ die Lösung von σ = A*(1 − σ) in (0, 1), A* die Laplace-Transformierte der
Zwischenankunftszeit (mittlere Dauer 1, also Rate μ = 1).
**Kingman** (G/G/1, Näherung für starke Auslastung): Wq ≈ ρ/(1 − ρ)·(ca² + cs²)/2.
**Allen-Cunneen** (G/G/c, Näherung): Wq ≈ Wq(M/M/c)·(ca² + cs²)/2 mit der Erlang-C-Wartezeit; für c = 1 gleich Kingman.

Die Erlang-C-Formel ist eine Kopie aus mmc-queue-demo (bewusst ohne Import zwischen Repos)."""

import math


def pk_wait(rho, cs2):
    """Pollaczek-Khinchine: mittlere Wartezeit der M/G/1-Schlange bei mittlerer Dauer 1: ρ(1 + cs²)/(2(1 − ρ))."""
    _check_rho(rho)
    return rho * (1.0 + cs2) / (2.0 * (1.0 - rho))


def mm1_wait(rho):
    """M/M/1 (cs² = 1): ρ/(1 − ρ)."""
    return pk_wait(rho, 1.0)


def kingman_wait(rho, ca2, cs2):
    """Kingman-Näherung der G/G/1-Wartezeit: ρ/(1 − ρ)·(ca² + cs²)/2."""
    _check_rho(rho)
    return rho / (1.0 - rho) * (ca2 + cs2) / 2.0


def erlang_b(c, a):
    """Erlang-B-Verlustwahrscheinlichkeit über die stabile Rekursion B_k = a·B_{k−1}/(k + a·B_{k−1})."""
    b = 1.0
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    return b


def erlang_c(c, a):
    """Erlang C: Wahrscheinlichkeit zu warten bei unendlicher Schlange, C = B/(1 − ρ(1 − B)); nur für c > a."""
    rho = a / c
    if rho >= 1:
        raise ValueError("c ≤ a: keine stationäre Verteilung")
    b = erlang_b(c, a)
    return b / (1 - rho * (1 - b))


def mmc_wait(c, rho):
    """Mittlere Wartezeit der M/M/c-Schlange bei mittlerer Dauer 1 und Auslastung ρ je Spur: C(c, cρ)/(c(1 − ρ))."""
    _check_rho(rho)
    return erlang_c(c, c * rho) / (c * (1.0 - rho))


def allen_cunneen_wait(c, rho, ca2, cs2):
    """Allen-Cunneen-Näherung der G/G/c-Wartezeit: Wq(M/M/c)·(ca² + cs²)/2."""
    return mmc_wait(c, rho) * (ca2 + cs2) / 2.0


def approx_wait(c, rho, ca2, cs2):
    """Die passende Näherung: Kingman für eine Spur, Allen-Cunneen für mehrere (c = 1 liefert dasselbe)."""
    return allen_cunneen_wait(c, rho, ca2, cs2)


def balanced_h2_probability(scv):
    """Anteil p₁ der ersten Phase einer Hyperexponentialverteilung mit gleichen Phasenanteilen am Mittel (balanced means), Variationskoeffizient² `scv` ≥ 1."""
    return 0.5 * (1.0 + math.sqrt((scv - 1.0) / (scv + 1.0)))


def interarrival_lst(rho, ca2):
    """Laplace-Transformierte A*(s) der Zwischenankunftszeit mit Mittel 1/ρ und Variationskoeffizient² `ca2`: fest (0), Erlang-k (1/k), exponentiell (1), Hyperexponential (> 1)."""
    if ca2 == 0:
        return lambda s: math.exp(-s / rho)
    if ca2 < 1:
        k = round(1.0 / ca2)
        return lambda s: (k * rho / (k * rho + s)) ** k
    if ca2 == 1:
        return lambda s: rho / (rho + s)
    p1 = balanced_h2_probability(ca2)
    r1, r2 = 2.0 * p1 * rho, 2.0 * (1.0 - p1) * rho
    return lambda s: p1 * r1 / (r1 + s) + (1.0 - p1) * r2 / (r2 + s)


def gim1_sigma(rho, ca2):
    """σ der G/M/1-Schlange (Dauer exponentiell mit Rate 1): Lösung von σ = A*(1 − σ) in (0, 1), per Bisektion."""
    _check_rho(rho)
    lst = interarrival_lst(rho, ca2)
    lo, hi = 0.0, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if lst(1.0 - mid) - mid > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gim1_wait(rho, ca2):
    """Exakte Wartezeit der G/M/1-Schlange (Dauer exponentiell, Mittel 1): σ/(1 − σ). Für ca² = 1 gleich M/M/1."""
    sigma = gim1_sigma(rho, ca2)
    return sigma / (1.0 - sigma)


def exact_wait(c, rho, ca2, cs2):
    """Exakte Wartezeit, wo es eine Formel gibt (sonst None): M/G/1 (c = 1, ca² = 1) Pollaczek-Khinchine; G/M/1 (c = 1, cs² = 1); M/M/c (ca² = cs² = 1)."""
    if c == 1 and ca2 == 1.0:
        return pk_wait(rho, cs2)
    if c == 1 and cs2 == 1.0:
        return gim1_wait(rho, ca2)
    if ca2 == 1.0 and cs2 == 1.0:
        return mmc_wait(c, rho)
    return None


def relative_error(approx, reference):
    """Näherung gegen Bezugswert: approx/reference − 1; None, wenn der Bezug null ist."""
    return approx / reference - 1.0 if reference > 1e-12 else None


def customers_for_precision(rel_sd_single_run, n_run, target=0.05):
    """Kunden, die ein Lauf bräuchte, damit seine relative Standardabweichung `target` beträgt: n·(sd/target)² (Streuung ∝ 1/√n)."""
    return n_run * (rel_sd_single_run / target) ** 2


def to_minutes(x, mean_service_min=3.0):
    """Zeit in Abfertigungsdauern → Minuten."""
    return x * mean_service_min


def _check_rho(rho):
    if not 0 < rho < 1:
        raise ValueError("ρ muss in (0, 1) liegen")
