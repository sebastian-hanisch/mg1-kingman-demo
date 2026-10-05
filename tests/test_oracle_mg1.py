"""Orakel-Tests (unabhängiger Rechenweg): exakte Wartezeiten aus abgeschnittenen Markov-Ketten (Phasentyp-Verteilungen), aus Geburts-Todes-Gleichungen und aus der
eingebetteten Kette von D/M/1 gegen Pollaczek-Khinchine, G/M/1, Erlang C und die Simulation; die Kiefer-Wolfowitz-Schleife gegen eine Brute-Force-FIFO-Rechnung."""

import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")
spla = pytest.importorskip("scipy.sparse.linalg")

import mg1_formulas as F  # noqa: E402
import mg1_simulation as S  # noqa: E402


def _stationary(rows, cols, vals, n):
    q = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    q = q - sp.diags(np.asarray(q.sum(axis=1)).ravel())
    a = sp.vstack([sp.csr_matrix(np.ones((1, n))), q.T.tocsr()[1:]]).tocsc()
    b = np.zeros(n)
    b[0] = 1.0
    return spla.spsolve(a, b)


def _phase_type(scv):
    """(Anfangsverteilung, Ratenmatrix, Austrittsraten) mit Mittel 1: Erlang-k (scv = 1/k), exponentiell, Hyperexponential mit gleichen Phasenanteilen am Mittel
    (p/μ je Phase 1/2, Anteil p₁ aus der zweiten Momentengleichung 1/(2p₁) + 1/(2p₂) = 1 + scv gelöst, nicht aus der Formel der Demo)."""
    if scv < 1:
        k = round(1 / scv)
        T = -k * np.eye(k) + k * np.eye(k, k=1)
        t = np.zeros(k)
        t[-1] = k
        return np.eye(k)[0], T, t
    if scv == 1:
        return np.array([1.0]), np.array([[-1.0]]), np.array([1.0])
    from scipy.optimize import brentq
    p = 1 - brentq(lambda q: 1 / (2 * q) + 1 / (2 * (1 - q)) - 1 - scv, 1e-9, 0.5 - 1e-12)
    return np.array([p, 1 - p]), np.diag([-2 * p, -2 * (1 - p)]), np.array([2 * p, 2 * (1 - p)])


def _m_ph_1_wait(rho, scv_s, nmax=200):
    """M/PH/1: Zustand (n, Phase des Kunden in Abfertigung); Wq = (L − P(besetzt))/λ (Little)."""
    alpha, T, t = _phase_type(scv_s)
    k = len(alpha)
    idx = lambda n, i: 0 if n == 0 else 1 + (n - 1) * k + i
    rows, cols, vals = [], [], []
    for i in range(k):
        rows.append(0); cols.append(idx(1, i)); vals.append(rho * alpha[i])
    for n in range(1, nmax + 1):
        for i in range(k):
            s = idx(n, i)
            if n < nmax:
                rows.append(s); cols.append(idx(n + 1, i)); vals.append(rho)
            for j in range(k):
                if j != i and T[i, j] > 0:
                    rows.append(s); cols.append(idx(n, j)); vals.append(T[i, j])
            for j in range(k):
                if t[i] * (alpha[j] if n > 1 else 1.0) > 0 and (n > 1 or j == 0):
                    rows.append(s); cols.append(idx(n - 1, j) if n > 1 else 0); vals.append(t[i] * (alpha[j] if n > 1 else 1.0))
    pi = _stationary(rows, cols, vals, 1 + nmax * k)
    length = sum(n * pi[idx(n, i)] for n in range(1, nmax + 1) for i in range(k))
    return (length - (1 - pi[0])) / rho


def _ph_m_1_wait(rho, scv_a, nmax=200):
    """PH/M/1: Zustand (n, Phase der laufenden Zwischenankunftszeit)."""
    alpha, T, t = _phase_type(scv_a)
    T, t = T * rho, t * rho
    k = len(alpha)
    idx = lambda n, i: n * k + i
    rows, cols, vals = [], [], []
    for n in range(nmax + 1):
        for i in range(k):
            s = idx(n, i)
            for j in range(k):
                if j != i and T[i, j] > 0:
                    rows.append(s); cols.append(idx(n, j)); vals.append(T[i, j])
                if n < nmax and alpha[j] > 0:
                    rows.append(s); cols.append(idx(n + 1, j)); vals.append(t[i] * alpha[j])
            if n >= 1:
                rows.append(s); cols.append(idx(n - 1, i)); vals.append(1.0)
    pi = _stationary(rows, cols, vals, (nmax + 1) * k)
    length = sum(n * pi[idx(n, i)] for n in range(nmax + 1) for i in range(k))
    return (length - (1 - sum(pi[idx(0, i)] for i in range(k)))) / rho


def _mmc_birth_death(c, rho, nmax=1500):
    w = np.ones(nmax + 1)
    for n in range(1, nmax + 1):
        w[n] = w[n - 1] * c * rho / min(n, c)
    pi = w / w.sum()
    ns = np.arange(nmax + 1)
    return ((ns - c).clip(0) * pi).sum() / (c * rho), pi[c:].sum()


def _dm1_wait(rho, nmax=150):
    """D/M/1 über die eingebettete Kette der beim Ankunftszeitpunkt vorgefundenen Zahl: X' = max(X + 1 − D, 0), D ~ Poisson(1/ρ)."""
    from math import lgamma, log
    T = 1.0 / rho
    a = np.exp(-T + np.arange(nmax + 2) * log(T) - np.array([lgamma(k + 1) for k in range(nmax + 2)]))
    P = np.zeros((nmax + 1, nmax + 1))
    for x in range(nmax + 1):
        d = np.arange(x + 1)
        j = x + 1 - d
        ok = j <= nmax
        P[x, j[ok]] += a[d[ok]]
        P[x, 0] += 1 - a[: x + 1].sum()
    P[nmax, nmax] += 1 - P[nmax].sum()
    A = (P - np.eye(nmax + 1)).T
    A[0, :] = 1.0
    b = np.zeros(nmax + 1)
    b[0] = 1.0
    return float((np.arange(nmax + 1) * np.linalg.solve(A, b)).sum())


def test_oracle_itself_by_hand():
    assert _m_ph_1_wait(0.8, 1.0) == pytest.approx(4.0)                      # M/M/1: ρ/(1 − ρ)
    assert _m_ph_1_wait(0.8, 0.5) == pytest.approx(3.0)                      # M/E2/1: ρ(1 + ½)/(2(1 − ρ))
    assert _ph_m_1_wait(0.5, 1.0) == pytest.approx(1.0)
    assert _mmc_birth_death(2, 0.5)[0] == pytest.approx(1 / 3)               # M/M/2, a = 1
    assert _dm1_wait(0.5) == pytest.approx(0.2550, abs=5e-4)


@pytest.mark.parametrize("rho", [0.4, 0.8])
@pytest.mark.parametrize("scv", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_pollaczek_khinchine_and_gim1_against_markov_chains(rho, scv):
    assert F.pk_wait(rho, scv) == pytest.approx(_m_ph_1_wait(rho, scv), rel=1e-6)
    assert F.gim1_wait(rho, scv) == pytest.approx(_ph_m_1_wait(rho, scv), rel=1e-6)


@pytest.mark.parametrize("rho", [0.5, 0.8])
def test_deterministic_arrivals_against_the_embedded_chain(rho):
    assert F.gim1_wait(rho, 0.0) == pytest.approx(_dm1_wait(rho), rel=1e-6)


@pytest.mark.parametrize("c,rho", [(2, 0.5), (4, 0.8), (4, 0.95), (7, 0.6)])
def test_erlang_c_against_birth_death_equations(c, rho):
    wq, pw = _mmc_birth_death(c, rho)
    assert F.mmc_wait(c, rho) == pytest.approx(wq, rel=1e-9) and F.erlang_c(c, c * rho) == pytest.approx(pw, abs=1e-12)


@pytest.mark.parametrize("c,rho,ca2,cs2", [(1, 0.8, 1.0, 4.0), (4, 0.7, 0.5, 2.0), (3, 0.9, 4.0, 0.25), (2, 0.6, 0.0, 1.0), (4, 0.8, 1.0, 1.0)])
def test_kiefer_wolfowitz_loop_against_brute_force_fifo(c, rho, ca2, cs2):
    """Selbe Zufallsströme, aber FIFO per Argmin über ein Feld freier Zeiten statt Heap; Warteanteil, Mittel und Auslastung nach Definition."""
    n, seed, warm_fraction = 3000, 7, 0.05
    ra, rs = S.streams(seed)
    arrive, serve = S.make_sampler(1.0 / (c * rho), ca2, ra), S.make_sampler(1.0, cs2, rs)
    gaps = np.array([arrive() for _ in range(n)])
    svcs = np.array([serve() for _ in range(n)])
    t = np.cumsum(gaps)
    free = np.zeros(c)
    waits = np.zeros(n)
    for k in range(n):
        j = int(np.argmin(free))
        start = max(t[k], free[j])
        waits[k] = start - t[k]
        free[j] = start + svcs[k]
    warm = int(warm_fraction * n)
    res = S.simulate(c, rho, ca2, cs2, n, seed, warm_fraction=warm_fraction)
    assert res.n == n - warm
    assert res.mean_wait == pytest.approx(waits[warm:].mean(), abs=1e-10)
    assert res.wait_prob == pytest.approx((waits[warm:] > 0).mean())
    assert res.utilisation == pytest.approx(svcs[warm:].sum() / (c * (t[-1] - t[warm])), rel=1e-10)


@pytest.mark.parametrize("cs2", [0.25, 4.0])
def test_simulation_against_markov_chain_waits(cs2):
    exact = _m_ph_1_wait(0.7, cs2)
    runs = [S.simulate(1, 0.7, 1.0, cs2, 80_000, 40 + 13 * r).mean_wait for r in range(4)]
    se = np.std(runs, ddof=1) / 2
    assert abs(np.mean(runs) - exact) < max(4 * se, 0.03 * exact)
