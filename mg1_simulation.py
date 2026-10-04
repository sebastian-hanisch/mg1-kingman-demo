"""Simulation eines Gates mit c Spuren, einer FIFO-Schlange und beliebiger Streuung von Zwischenankunftszeit und Abfertigungsdauer (G/G/c).

Streuung über den Variationskoeffizienten² (scv) bei Mittel 1: 0 fest, 1/k Erlang-k, 1 exponentiell, über 1 Hyperexponential mit gleichen Phasenanteilen am
Mittel (balanced means). Kein Ereignisheap nötig: bei FIFO und c gleichen Spuren gilt die Kiefer-Wolfowitz-Rekursion über die Zeitpunkte, zu denen jede Spur frei
wird (Heap): der Kunde startet bei max(Ankunft, früheste freie Spur).

Aufbau nach Einheiten: `Sampler`-Fabrik (`make_sampler`), `simulate` (Schleife), `lindley_mean_wait` (Lindley-Rekursion für c = 1, derselbe Pfad aus denselben Strömen,
also eine exakte Gegenprobe). Zufall nur über übergebene `SplitMix64`-Ströme (Zwischenankunft, Dauer)."""

import heapq
import math
from dataclasses import dataclass

_MASK = (1 << 64) - 1


class SplitMix64:
    """Kleiner, gut gemischter 64-Bit-Zufallsgenerator (Vigna); reine Ganzzahl-Arithmetik."""

    def __init__(self, seed):
        self.state = seed & _MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def uniform(self):
        """Gleichverteilt auf [0, 1) mit 53 Bit."""
        return (self.next() >> 11) * (1.0 / (1 << 53))

    def expovariate(self, rate):
        """Exponentiell mit Mittel 1/rate (Inversion; 1 − u liegt in (0, 1], der Logarithmus ist endlich)."""
        return -math.log(1.0 - self.uniform()) / rate


def streams(seed):
    """Die zwei Zufallsströme eines Laufs: Zwischenankunft und Abfertigungsdauer."""
    return SplitMix64(seed), SplitMix64(seed + 99_991)


def balanced_h2_probability(scv):
    """Anteil p₁ der ersten Phase der Hyperexponentialverteilung mit gleichen Phasenanteilen am Mittel (scv ≥ 1)."""
    return 0.5 * (1.0 + math.sqrt((scv - 1.0) / (scv + 1.0)))


def make_sampler(mean, scv, rng):
    """Funktion ohne Argument, die Werte mit dem Mittel `mean` und dem Variationskoeffizienten² `scv` zieht: fest, Erlang-k, exponentiell oder Hyperexponential."""
    if scv == 0:
        return lambda: mean
    if scv == 1:
        return lambda: rng.expovariate(1.0 / mean)
    if scv < 1:
        k = round(1.0 / scv)
        return lambda: sum(rng.expovariate(k / mean) for _ in range(k))
    p1 = balanced_h2_probability(scv)
    rate1, rate2 = 2.0 * p1 / mean, 2.0 * (1.0 - p1) / mean

    def draw():
        return rng.expovariate(rate1) if rng.uniform() < p1 else rng.expovariate(rate2)
    return draw


@dataclass
class SimResult:
    c: int
    rho: float
    ca2: float
    cs2: float
    n: int                      # ausgewertete Kunden
    mean_wait: float            # mittlere Wartezeit (Abfertigungsdauern)
    wait_prob: float            # Anteil der Kunden, die warten müssen
    utilisation: float          # mittlere Auslastung je Spur
    mean_interarrival: float    # beobachtete mittlere Zwischenankunftszeit (Soll 1/(c·ρ))
    mean_service: float         # beobachtete mittlere Dauer (Soll 1)


def simulate(c, rho, ca2, cs2, n_customers, seed, warm_fraction=0.05, rngs=None, samplers=None):
    """Ein Lauf über `n_customers` Kunden; die ersten `warm_fraction` davon werden nicht ausgewertet (Start leer). Auslastung ρ je Spur heißt: mittlere Zwischenankunftszeit
    1/(c·ρ), mittlere Dauer 1. `samplers` = (Ankunft, Dauer) ersetzt die Fabrik (für Mini-Instanzen von Hand)."""
    if samplers is None:
        ra, rs = rngs if rngs is not None else streams(seed)
        arrive, serve = make_sampler(1.0 / (c * rho), ca2, ra), make_sampler(1.0, cs2, rs)
    else:
        arrive, serve = samplers
    free = [0.0] * c
    heapq.heapify(free)
    t, warm = 0.0, int(warm_fraction * n_customers)
    wait_sum, waited, count, busy_time, inter_sum, svc_sum = 0.0, 0, 0, 0.0, 0.0, 0.0
    t_first_eval = 0.0
    for i in range(n_customers):
        gap = arrive()
        t += gap
        earliest = heapq.heappop(free)
        start = t if t >= earliest else earliest
        s = serve()
        heapq.heappush(free, start + s)
        if i == warm:
            t_first_eval = t
        if i >= warm:
            count += 1
            w = start - t
            wait_sum += w
            if w > 0:
                waited += 1
            busy_time += s
            inter_sum += gap
            svc_sum += s
    horizon = max(t - t_first_eval, 1e-12)
    return SimResult(c=c, rho=rho, ca2=ca2, cs2=cs2, n=count, mean_wait=wait_sum / count, wait_prob=waited / count,
                     utilisation=busy_time / (c * horizon), mean_interarrival=inter_sum / count, mean_service=svc_sum / count)


def lindley_mean_wait(rho, ca2, cs2, n_customers, seed, warm_fraction=0.05, rngs=None):
    """Lindley-Rekursion W_{n+1} = max(0, W_n + S_n − A_{n+1}) für eine Spur aus denselben Strömen wie `simulate` (c = 1): liefert denselben Pfad, also
    dieselbe mittlere Wartezeit (bis auf Rundung)."""
    ra, rs = rngs if rngs is not None else streams(seed)
    arrive, serve = make_sampler(1.0 / rho, ca2, ra), make_sampler(1.0, cs2, rs)
    warm = int(warm_fraction * n_customers)
    arrive()                                   # erste Zwischenankunft (der erste Kunde wartet nie)
    w, total, count = 0.0, 0.0, 0
    for i in range(n_customers):
        if i >= warm:
            total += w
            count += 1
        w = max(0.0, w + serve() - arrive())
    return total / count
