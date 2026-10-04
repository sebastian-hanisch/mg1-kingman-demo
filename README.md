# M/G/1 und Kingman – Streuung kostet Wartezeit (Streamlit-Demo)

**[→ Demo live ausprobieren](https://sebastianhanisch-mg1-kingman-demo.streamlit.app/)**

Interaktive Demo zur **Wartezeit bei beliebiger Streuung** von Ankünften und Abfertigung am Terminal-Gate. **Zehntes Stück der Konzepte-Linie „Warteschlangentheorie und
Simulation“** im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning): ein Verfahren, ein wachsendes Beispiel, jedes
Folgestück hebt genau eine Annahme auf.

Bisher war die Abfertigung exponentiell und die Ankunft Poisson. In [mm1-queue-demo](https://github.com/sebastian-hanisch/mm1-queue-demo) (Stück 1) halbierte feste Dauer die
Wartezeit schon als Gegenbeispiel; in [erlang-b-demo](https://github.com/sebastian-hanisch/erlang-b-demo) (Stück 8) hing der **Verlust** gar nicht von der Streuung ab. Das **Warten**
tut es, und zwar **linear**: Pollaczek-Khinchine sagt Wq = ρ/(1 − ρ) · (1 + cs²)/2 · mittlere Dauer, Kingman ergänzt die Streuung der Ankünfte: (ca² + cs²)/2.

## Kernfrage

Wie stark hängt die Wartezeit von der Streuung ab, wie gut sind Kingman und Allen-Cunneen als Näherungen (und wo nicht), und was kostet eine genaue Simulation?

## Modell und Methodik

- **Gate:** c Spuren (1 oder 4), eine FIFO-Schlange, unendlicher Warteraum, Auslastung ρ je Spur (50 bis 95 %), mittlere Abfertigung 3 min. Streuung über den quadrierten
  Variationskoeffizienten bei Mittel 1: **cs²** der Dauer = 0 (fest), 0.25 (Erlang-4), 1 (exponentiell), 4 (Hyperexponential mit gleichen Phasenanteilen am Mittel); **ca²** der
  Ankünfte = 0 (glatt wie Termine), 1 (Poisson), 4 (stoßweise).
- **Formeln** (`mg1_formulas.py`): **M/G/1, Pollaczek-Khinchine (exakt):** Wq = ρ(1 + cs²)/(2(1 − ρ)). **G/M/1 (exakt):** beliebige Ankünfte, exponentielle Dauer: Wq = σ/(1 − σ), σ = A*(1 − σ)
  mit der Laplace-Transformierten A* der Zwischenankunftszeit (fest, Erlang, exponentiell, Hyperexponential), per Bisektion gelöst. **Kingman (G/G/1, Näherung):** Wq ≈ ρ/(1 − ρ)·(ca² + cs²)/2.
  **Allen-Cunneen (G/G/c, Näherung):** Wq ≈ Wq(M/M/c)·(ca² + cs²)/2 mit der Erlang-C-Wartezeit; für eine Spur gleich Kingman.
- **Simulation** (`mg1_simulation.py`): Kiefer-Wolfowitz-Rekursion über die Zeitpunkte, zu denen die Spuren frei werden (Heap); SplitMix64 mit getrennten Strömen für Zwischenankunft und Dauer; die
  ersten 5 % der Kunden werden nicht ausgewertet.
- **Gegenproben:** (1) für eine Spur liefert die **Lindley-Rekursion** aus denselben Strömen denselben Pfad wie die Kiefer-Wolfowitz-Simulation (Gleichheit bis auf Rundung, fünf Kombinationen);
  (2) Simulation gegen die exakten Formeln (Pollaczek-Khinchine für alle vier cs², G/M/1 für drei ca², Erlang C für vier Spuren samt P(Warten)); (3) die Laplace-Transformierten haben das Mittel 1/ρ und
  die richtige Streuung (numerische Ableitungen); (4) die Sampler haben Mittel 1 und die verlangte Streuung; (5) von Hand gerechnete Mini-Instanzen (eine und zwei Spuren, Einschwingen).
- **Vorgerechnete Studie** (`generate_precomputed.py` → `precomputed_sweep.json`, rund dreieinhalb Minuten parallel): 2 Spurzahlen × 3 Auslastungen (50, 80, 95 %) × 3 ca² × 4 cs², je 6 Läufe à 1 500 000 Kunden;
  dazu eine **Aufwandsmessung** (Poisson-Ankünfte, je 48 Läufe à 100 000 Kunden), weil die Streuung eines Laufs aus nur 6 Läufen zu unsicher ist.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Wartezeiten in Abfertigungsdauern, wo nicht Minuten dasteht.

| Frage | Befund |
|---|---|
| Wie stark hängt Warten von der Streuung der Dauer ab? | Eine Spur, Poisson-Ankünfte, Auslastung 80 %: Pollaczek-Khinchine **2.0 / 2.5 / 4.0 / 10.0** für cs² = 0 / 0.25 / 1 / 4; Simulation 2.004 / 2.500 / 3.987 / 9.878, also das **0.50- / 0.62- / 1.00- / 2.47-Fache** von M/M/1. |
| Stimmt die Simulation? | In allen 21 Zellen mit exakter Formel (Pollaczek-Khinchine, G/M/1, Erlang C) weicht sie höchstens **3.7 %** ab, im Mittel 0.6 %. |
| Ist Kingman bei Poisson-Ankünften exakt? | Ja, für eine Spur ist es Pollaczek-Khinchine; die Simulation weicht in allen zwölf Zellen höchstens 3.9 % ab (Rauschen). |
| Und bei anderen Ankünften? | Eine Spur, Auslastung 50 / 80 / 95 %: glatte Ankünfte, exponentielle Dauer (D/M/1): Kingman **+96 / +18 / +2.4 %**; stoßweise Ankünfte, feste Dauer +52 / +7.6 / +0.7 %; stoßweise Ankünfte, exponentielle Dauer +16 / +1.6 / −0.1 %. **Der Fehler sinkt mit der Auslastung** (Schwerverkehr). |
| D/M/1 gegen die exakte Formel? | Simulation 0.255 / 1.694 / 9.281 gegen exakt (G/M/1) 0.255 / 1.693 / 9.172 bei Auslastung 50 / 80 / 95 %. |
| Wie gut ist Allen-Cunneen (vier Spuren)? | **Relativ** schlecht bei geringer Last: bis +2552 % bei 50 % (D/M/4: 0.043 gegen 0.010), bis 77 % bei 80 %; **absolut** winzig: höchstens 0.3 min bei 50 %, 0.9 min bei 80 %. Bei 95 % liegen alle Zellen mit Wartezeit innerhalb von ±11.1 %. M/M/4 (Erlang C) stimmt auf −0.1 / +0.3 / −1.7 %. |
| Richtung des Fehlers bei mehreren Spuren? | Auslastung 50 %: glatte Ankünfte **+323 %** (exponentielle Dauer), stoßweise Ankünfte **−26 %**, stoßweise Dauer bei Poisson-Ankünften +30 %: Allen-Cunneen überschätzt Glätte und unterschätzt Stöße. |
| Was kostet Genauigkeit? | Kunden für ±5 % (eine Spur, Poisson-Ankünfte, cs² = 0 / 0.25 / 1 / 4), in Tausend: bei Auslastung 50 % **8 / 9 / 15 / 41**, bei 80 % **18 / 30 / 41 / 145**, bei 95 % **305 / 361 / 800 / 1546**. Streuende Dauer (cs² = 4) braucht bei 80 % das 8.2-Fache, bei 95 % das 5.1-Fache an Kunden gegenüber fester Dauer. |
| Wie verlässlich sind die Zahlen? | Die relative Streuung eines 1.5-Mio-Laufs beträgt höchstens 8.2 % (Mittel aus 6 Läufen: 3.3 %), meist unter 3 %. |
| Standardlauf der App? | Eine Spur, 80 %, Poisson, cs² = 4, 150 000 Kunden, Seed 35: **32.01 min** (exakt 30.00 min, +6.7 %, das 2.67-Fache von M/M/1), 81.2 % der Kunden warten. |

## Befunde und Korrekturen gegenüber der Vorab-Messreihe

- **Die Streuung aus 6 Läufen ist verrauscht.** Die erste Fassung der Aufwandsrechnung benutzte die 6 Läufe der Hauptstudie und lieferte bei Auslastung 80 % für feste Dauer 4.0·10⁴ und für Erlang-4
  nur 1.5·10⁴ Kunden: das ist unplausibel (Erlang-4 streut mehr als feste Dauer und brauchte trotzdem weniger Kunden). Die Streuung eines Laufs braucht viele Wiederholungen; die Aufwandsmessung
  mit 48 Läufen liefert die monotone Reihe 18 / 30 / 41 / 145 (Tausend).
- **Die Vorab-Messreihe nannte „+95 %“ für D/M/1 bei Auslastung 50 %** (aus einer Simulation); der exakte G/M/1-Wert ergibt +96 % (0.500 gegen 0.255).
- **Bei Auslastung 95 % und cs² = 4 lag die Simulation der Vorab-Messreihe 5 % unter der Formel**, trotz kleinem Standardfehler (schwerer Schwanz). In der Studie sind es −3.7 % (stark streuende Dauer, Poisson-Ankünfte, eine Spur,
  Auslastung 95 %): auch dort die Simulation unter dem exakten Wert, mit deutlicher Streuung eines Laufs.
- **Der Standardlauf liegt 6.7 % über dem exakten Wert**: ein Lauf mit 150 000 Kunden und stark streuender Dauer streut um Zehntel; die Studie mittelt sechs lange Läufe.

## Ehrliche Grenzen

- Zwischenankünfte und Dauern sind unabhängig und unkorreliert; Kingman kennt nur die Varianz. Wellen, Fähren-Pulks und Serien langer Abfertigungen (Autokorrelation) machen die Wartezeit länger.
- Nur die **mittlere Wartezeit**: Quantile (90 % der Lkw warten höchstens …) brauchen die ganze Verteilung und sind nicht abgebildet.
- Allen-Cunneen ist bei geringer Last relativ ungenau; bessere Näherungen und eine exakte Lösung für G/G/c sind in dieser Demo nicht gerechnet.
- Die Streuung nimmt nur vier (Dauer) bzw. drei (Ankünfte) Werte an; der Hyperexponentialfall hat feste Phasenanteile am Mittel (balanced means); andere Verteilungen gleicher Streuung (etwa lognormal)
  sind nicht gerechnet. Die exakten G/M/1-Werte gelten für exponentielle Dauer.
- Zwei Spurzahlen (1 und 4); die App zeigt Studienabschnitte für die nächste Auslastung (50, 80, 95 %) und sagt es.
- Der Live-Lauf ist kurz (150 000 Kunden) und streut bei stark streuender Dauer um Zehntel des Werts.
- Unendlicher Warteraum und unendliche Geduld; mit Abwanderung oder Stellplätzen siehe Stück 4 und 8.

## Verwandte Demos im Portfolio

- [`markov-queue-demo`](https://github.com/sebastian-hanisch/markov-queue-demo) (Zusatzstück: Phasen-Ketten für Erlang-Dauern ergeben genau die Pollaczek-Khinchine-Wartezeit).
- [`mm1-queue-demo`](https://github.com/sebastian-hanisch/mm1-queue-demo) (Stück 1): eine Spur, feste Dauer halbiert die Wartezeit.
- [`mmc-queue-demo`](https://github.com/sebastian-hanisch/mmc-queue-demo) (Stück 3): mehrere Spuren, Erlang C als Basis der Näherung.
- [`erlang-b-demo`](https://github.com/sebastian-hanisch/erlang-b-demo) (Stück 8): der Verlust ist unempfindlich gegen die Streuung, das Warten nicht.
- [`output-analysis-demo`](https://github.com/sebastian-hanisch/output-analysis-demo) (Stück 2): Konfidenzintervalle und der Aufwand für Genauigkeit.
- [`truck-appointment-demo`](https://github.com/sebastian-hanisch/truck-appointment-demo): Terminvergabe erzeugt glatte Ankünfte (ca² < 1); dort liegt der Gewinn, den Kingman hier quantifiziert.

## Bewusst nicht umgesetzt

Jede dieser Annahmen hebt ein Folgestück der Linie auf:

| Annahme | Folgestück |
|---|---|
| Zwischenankünfte und Dauern unkorreliert | [Zeitvariable Ankünfte](https://github.com/sebastian-hanisch/time-varying-arrivals-demo) |
| Alle Lkw gleich wichtig | [Prioritätsklassen](https://github.com/sebastian-hanisch/priority-queue-demo) |
| Ein Gate | [Jackson-Netze](https://github.com/sebastian-hanisch/jackson-network-demo) |
| Geduldige Lkw, unbegrenzter Warteraum | [Erlang A](https://github.com/sebastian-hanisch/erlang-a-demo), [Erlang B](https://github.com/sebastian-hanisch/erlang-b-demo) |

Kein Folgestück: Verteilung der Wartezeit (Quantile), bessere Näherungen für mehrere Spuren.

## Tests

120 Tests, rund 40 Sekunden: Pollaczek-Khinchine von Hand, Kingman (symmetrisch, gleich Pollaczek-Khinchine bei ca² = 1), Allen-Cunneen (gleich Kingman für eine Spur, gleich Erlang C bei Poisson/exponentiell),
die Laplace-Transformierten (Mittel und Streuung), G/M/1 (gleich M/M/1 bei Poisson, Fixpunkt, D/M/1 von Hand, Monotonie in ca², Kingman-Fehler sinkt mit der Last), die Sampler, Mini-Instanzen von Hand, die
Lindley-Rekursion als exakte Gegenprobe, Simulation gegen alle exakten Formeln, Vollständigkeit der vorgerechneten Datei, Presets und Permalink, Diagramme (gesperrte Achsen), AppTest-Rauchtests mit festem Würfel-Seed,
der Smoke-Test der Portfolio-Vorlage, ein Quelltext-Test gegen Satz-Komma-Fehler und `test_claims.py` für jede Zahl dieser README.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `mg1_formulas.py` | Pollaczek-Khinchine, Kingman, Allen-Cunneen, G/M/1, Erlang C |
| `mg1_simulation.py` | Sampler, Kiefer-Wolfowitz, Lindley |
| `mg1_evaluation.py` | Live-Lauf, Studienzellen, Aufwand |
| `generate_precomputed.py` | rechnet die Studie vor → `precomputed_sweep.json` |
| `mg1_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `mg1_presets.py`, `mg1_constants.py` | Presets, Permalink, Grenzen |
| `tests/` | siehe oben |

## Literatur

- Pollaczek, F. (1930): Mittlere-Wartezeit-Formel für die M/G/1-Schlange; Khinchine fand sie wenig später unabhängig (Pollaczek-Khinchine-Formel); Titel und Quelle der Originale nicht einzeln belegt.
- Kingman, J. F. C. (1961): The single server queue in heavy traffic. *Mathematical Proceedings of the Cambridge Philosophical Society* 57(4), 902–904 (die Näherung für G/G/1, nach ihm benannt).
- Allen-Cunneen-Näherung für G/G/c (Wartezeit der M/M/c-Schlange mal (ca² + cs²)/2); Quelle nicht einzeln belegt.

## Lokal ausführen

```
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -v`. Studie neu rechnen: `python generate_precomputed.py`.

Gebaut mit Streamlit und Plotly.
