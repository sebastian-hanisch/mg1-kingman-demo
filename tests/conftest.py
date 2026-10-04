import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class ScriptedRng:
    """Liefert vorgegebene Werte statt Zufall (Mini-Instanzen von Hand gerechnet). `expovariate(rate)` ignoriert die Rate und gibt der
    Reihe nach die Werte zurück, `uniform()` ebenso aus einer eigenen Liste."""

    def __init__(self, exp_values=(), uniform_values=()):
        self.exp_values, self.uniform_values = list(exp_values), list(uniform_values)
        self.n_exp = self.n_uniform = 0

    def expovariate(self, rate):
        v = self.exp_values[self.n_exp]
        self.n_exp += 1
        return v

    def uniform(self):
        v = self.uniform_values[self.n_uniform]
        self.n_uniform += 1
        return v


@pytest.fixture
def two_customers_one_lane():
    """Eine Spur, vier Kunden. Zwischenankünfte 1.0 / 0.5 / 0.5 / 3.0 (Ankünfte bei 1.0, 1.5, 2.0, 5.0), Dauern 2.0 / 1.0 / 0.5 / 0.2.
    Von Hand: Kunde 1 startet bei 1.0 (Ende 3.0, Wartezeit 0); Kunde 2 (1.5) wartet bis 3.0 (Wartezeit 1.5, Ende 4.0); Kunde 3 (2.0) wartet bis 4.0
    (Wartezeit 2.0, Ende 4.5); Kunde 4 (5.0) findet die Spur frei (Wartezeit 0). Mittlere Wartezeit (0 + 1.5 + 2.0 + 0)/4 = 0.875."""
    return ScriptedRng(exp_values=[1.0, 0.5, 0.5, 3.0]), ScriptedRng(exp_values=[2.0, 1.0, 0.5, 0.2])
