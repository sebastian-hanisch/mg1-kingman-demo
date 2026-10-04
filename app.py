"""M/G/1 und Kingman - Streuung kostet Wartezeit - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zehntes Stück der Konzepte-Linie "Warteschlangentheorie und Simulation": Bisher war die Abfertigung exponentiell und die Ankunft Poisson. Hier haben beide
beliebige Streuung. Die Wartezeit hängt davon linear ab (Pollaczek-Khinchine für M/G/1 exakt, Kingman für G/G/1 und Allen-Cunneen für mehrere Spuren als
Näherungen), im Gegensatz zum Verlust aus Stück 8. Die Demo zeigt Formeln gegen Simulation, die Genauigkeit der Näherungen und was Genauigkeit in der
Simulation kostet. Siehe README für die Einordnung in die Linie.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import mg1_constants as C
import mg1_formulas as F
from mg1_evaluation import (approx_error, cell_mean, customers_needed, effort_cell, live_report, load_precomputed, nearest, study_cell)
from mg1_presets import (apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed,
                         sync_query_params)
from mg1_visualization import (build_compare_chart, build_effort_chart, build_error_heatmap, build_heavy_traffic_chart, build_pk_chart, ca2_short,
                               cs2_short)

st.set_page_config(page_title="M/G/1 und Kingman – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _precomputed():
    return load_precomputed()


@st.cache_data(show_spinner=False)
def _live(c, rho_pct, ca2, cs2, seed):
    return live_report(c, rho_pct, ca2, cs2, seed)


def _min(x):
    """Zeit in Abfertigungsdauern → Text in Minuten."""
    return f"{F.to_minutes(x):.2f} min"


st.title("📏 M/G/1 und Kingman: Streuung kostet Wartezeit")
st.markdown(
    """
Bisher war die Abfertigung **exponentiell** und die Ankunft **Poisson**. Echte Gates sind regelmäßiger (feste Abfertigung, Termine) oder unregelmäßiger (lange
Ausreißer, Pulks). In Stück 8 hing der **Verlust** nicht von der Streuung ab; das **Warten** tut es, und zwar **linear**: Die Formel von **Pollaczek und Khinchine** sagt für
eine Spur mit Poisson-Ankünften Wq = ρ/(1 − ρ) · **(1 + cs²)/2** · mittlere Dauer. Feste Dauer halbiert die Wartezeit gegenüber exponentieller, eine Dauer mit cs² = 4 verfünffacht
sie. **Kingman** ergänzt die Streuung der Ankünfte: (ca² + cs²)/2, symmetrisch, als Näherung für starke Auslastung.
"""
)
st.caption(
    "Zehntes Stück der Linie „Warteschlangentheorie und Simulation“, aufbauend auf "
    "[mm1-queue-demo](https://sebastianhanisch-mm1-queue-demo.streamlit.app/) (dort schon: feste Dauer halbiert die Wartezeit) und "
    "[mmc-queue-demo](https://sebastianhanisch-mmc-queue-demo.streamlit.app/) (mehrere Spuren, Erlang C). Jedes Folgestück hebt eine der Annahmen "
    "unter „Wo die Annahmen enden“ auf."
)

with st.expander("So funktionieren die Formeln", expanded=True):
    st.markdown(
        """
- **Streuung:** cs² (Dauer) und ca² (Zwischenankunftszeit) sind quadrierte Variationskoeffizienten bei Mittel 1: 0 = fest, 0.25 = Erlang-4, 1 = exponentiell, 4 = stark streuend
  (Hyperexponential). Ankünfte: glatt (ca² = 0, wie Termine), Poisson (1), stoßweise (4).
- **M/G/1, Pollaczek-Khinchine (exakt):** Wq = ρ(1 + cs²)/(2(1 − ρ)) in Abfertigungsdauern, aus dem zweiten Moment E[S²] = (1 + cs²) der Dauer.
- **G/M/1 (exakt):** beliebige Ankünfte, exponentielle Dauer: Wq = σ/(1 − σ) mit σ = A*(1 − σ), A* = Laplace-Transformierte der Zwischenankunftszeit.
- **Kingman (G/G/1, Näherung):** Wq ≈ ρ/(1 − ρ)·(ca² + cs²)/2, für ca² = 1 identisch mit Pollaczek-Khinchine.
- **Allen-Cunneen (G/G/c, Näherung):** Wq ≈ Wq(M/M/c)·(ca² + cs²)/2, für eine Spur gleich Kingman.
- **Simulation:** Kiefer-Wolfowitz-Rekursion über die Zeitpunkte, zu denen die Spuren frei werden; Streuung über Erlang-k, Hyperexponential (gleiche Phasenanteile am Mittel) oder fest.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESET_ORDER))
for i, name in enumerate(C.PRESET_ORDER):
    with preset_cols[i]:
        st.button(name, key=f"preset_{name}", width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name] or None)

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    rho_pct = st.slider("Auslastung ρ je Spur", *bounds("rho_slider"), step=C.RHO_PCT_STEP, key="rho_slider", format="%d %%",
                        help="Anteil der Zeit, in der eine Spur im Mittel beschäftigt ist.")
    c = st.select_slider("Spuren c", options=C.C_OPTIONS, key="c_select", help="1 = Pollaczek-Khinchine und Kingman, 4 = Allen-Cunneen.")
    ca2 = st.select_slider("Streuung der Ankünfte ca²", options=C.CA2_OPTIONS, key="ca2_select", format_func=lambda v: C.CA2_LABELS[v],
                           help="Variationskoeffizient² der Zwischenankunftszeit: glatt wie bei Terminen, Poisson oder stoßweise.")
    cs2 = st.select_slider("Streuung der Dauer cs²", options=C.CS2_OPTIONS, key="cs2_select", format_func=lambda v: C.CS2_LABELS[v],
                           help="Variationskoeffizient² der Abfertigungsdauer bei gleichem Mittel von 3 min.")
    seed = st.number_input("Zufalls-Seed", min_value=bounds("seed_input")[0], max_value=bounds("seed_input")[1], step=1,
                           key="seed_input", help="Bestimmt alle Zufallszahlen der Simulation.")
    st.button("🎲 Neuen Lauf würfeln", on_click=randomize_seed)

rho_pct, c, seed = int(rho_pct), int(c), int(seed)
ca2, cs2 = float(ca2), float(cs2)
sync_query_params({"rho_slider": rho_pct, "c_select": c, "ca2_select": ca2, "cs2_select": cs2, "seed_input": seed})

pre = _precomputed()
study_rho = nearest(C.STUDY_RHO_PCT, rho_pct)
with st.spinner("Simuliere das Gate …"):
    live = _live(c, rho_pct, ca2, cs2, seed)
sim = live["sim"]

st.markdown("---")
st.markdown("## 📏 Wartezeit bei beliebiger Streuung")
st.caption(
    f"{c} Spur(en), Auslastung {rho_pct} %, Ankünfte {ca2_short(ca2)} (ca² = {ca2:g}), Dauer {cs2_short(cs2)} (cs² = {cs2:g}). Simuliert: {C.fmt_int(C.LIVE_CUSTOMERS)} Kunden "
    f"(die ersten 5 % nicht ausgewertet)."
)
r1 = st.columns(3)
r1[0].metric("Wartezeit (Simulation)", _min(live["wait_sim"]))
r1[1].metric("Näherung (Kingman bzw. Allen-Cunneen)", _min(live["approx"]),
             help="Wq(M/M/c)·(ca² + cs²)/2; für eine Spur Kingman. Bei ca² = 1 und einer Spur ist das Pollaczek-Khinchine und exakt.")
r1[2].metric("Exakter Wert", _min(live["exact"]) if live["exact"] is not None else "keine Formel",
             help="Pollaczek-Khinchine (eine Spur, Poisson-Ankünfte), G/M/1 (eine Spur, exponentielle Dauer) oder Erlang C (M/M/c)." if live["exact"] is not None
             else "Für diese Kombination von Ankunfts- und Dauerstreuung gibt es keine geschlossene Formel; die Näherung ist die beste Auskunft.")
r2 = st.columns(3)
r2[0].metric(f"M/M/{c} (Bezug)", _min(live["mm"]), help="Poisson-Ankünfte und exponentielle Dauer bei gleicher Auslastung.")
r2[1].metric("Simulation gegenüber M/M", f"{live['wait_sim'] / live['mm']:.2f}-fach", help="Das Verhältnis der Wartezeit zum Bezug mit cs² = ca² = 1.")
r2[2].metric("Kunden, die warten müssen (Simulation)", C.fmt_pct(sim.wait_prob, 1))

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Wartezeit über die Auslastung (M/G/1, exakt)**")
    st.plotly_chart(build_pk_chart(rho_pct, cs2, live["wait_sim"] if (c == 1 and ca2 == 1.0) else None), width="stretch", key=f"pk_{rho_pct}_{cs2}_{c}_{ca2}_{seed}")
with col_b:
    st.markdown("**Formeln gegen Simulation**")
    st.plotly_chart(build_compare_chart(live, c, ca2, cs2), width="stretch", key=f"cmp_{rho_pct}_{c}_{ca2}_{cs2}_{seed}")
note = ""
if live["exact"] is not None:
    note = f" Der exakte Wert liegt bei {_min(live['exact'])}, die Simulation weicht um {C.fmt_signed_pct(sim.mean_wait / live['exact'] - 1, 1)} ab (Zufall eines Laufs)."
st.info(
    f"Bei Auslastung {rho_pct} % wartet ein Kunde im Mittel **{_min(live['wait_sim'])}**; mit Poisson-Ankünften und exponentieller Dauer wären es {_min(live['mm'])}. "
    f"Die Näherung sagt {_min(live['approx'])}.{note}"
)
st.caption(
    f"Ein Live-Lauf ist kurz ({C.fmt_int(C.LIVE_CUSTOMERS)} Kunden): bei stark streuender Dauer und hoher Auslastung streut er um Zehntel des Werts. Die Studie unten mittelt je "
    f"{C.STUDY_REPS} Läufe à {C.fmt_int(C.STUDY_CUSTOMERS)} Kunden."
)

st.markdown("---")
st.subheader("📐 Wie genau sind Kingman und Allen-Cunneen?")
st.markdown(
    f"Fehler der Näherung gegen die Simulation (Näherung/Simulation − 1) für {c} Spur(en) bei Auslastung {study_rho} %: **rot** = die Näherung überschätzt, **blau** = sie unterschätzt. "
    f"Je Zelle {pre['study_reps']} Läufe à {C.fmt_int(pre['study_customers'])} Kunden."
)
col_c, col_d = st.columns(2)
with col_c:
    st.markdown(f"**Fehler bei Auslastung {study_rho} %**")
    st.plotly_chart(build_error_heatmap(pre, c, study_rho), width="stretch", key=f"heat_{c}_{study_rho}")
with col_d:
    st.markdown(f"**Fehler über die Auslastung (Dauer {cs2_short(cs2)})**")
    st.plotly_chart(build_heavy_traffic_chart(pre, c, cs2), width="stretch", key=f"heavy_{c}_{cs2}")
rows = []
for a2 in C.STUDY_CA2:
    for s2 in C.STUDY_CS2:
        cell = study_cell(pre, c, study_rho, a2, s2)
        err = approx_error(cell)
        ex = cell["exact"]
        rows.append(f"| {ca2_short(a2)} | {cs2_short(s2)} | {_min(cell_mean(cell))} | {_min(cell['approx'])} | {C.fmt_signed_pct(err) if err is not None else '–'} | "
                    f"{_min(ex) if ex is not None else 'keine Formel'} |")
st.markdown("| Ankünfte | Dauer | Simulation | Näherung | Fehler der Näherung | exakt |\n|---|---|---|---|---|---|\n" + "\n".join(rows))
d_cell = study_cell(pre, 1, 50, 0.0, 1.0)
d_hi = study_cell(pre, 1, 95, 0.0, 1.0)
st.info(
    f"Die Näherung ist für Poisson-Ankünfte (ca² = 1) bei einer Spur exakt; sonst überschätzt sie meist. Bei glatten Ankünften und exponentieller Dauer (D/M/1, eine Spur) "
    f"liegt sie bei Auslastung 50 % {C.fmt_signed_pct(approx_error(d_cell))} und bei 95 % {C.fmt_signed_pct(approx_error(d_hi))} neben der Simulation: Kingman ist eine Näherung "
    "für **starke Auslastung**."
)

st.markdown("---")
st.subheader("🔬 Was kostet Genauigkeit in der Simulation?")
st.markdown(
    f"Kunden, die eine Simulation für eine relative Streuung von 5 % ihrer mittleren Wartezeit bräuchte ({c} Spur(en), Poisson-Ankünfte), hochgerechnet aus der Streuung von "
    f"{pre['effort_reps']} Läufen à {C.fmt_int(pre['effort_customers'])} Kunden. Hohe Auslastung und streuende Dauer verlangen viel mehr Kunden."
)
st.plotly_chart(build_effort_chart(pre, c), width="stretch", key=f"effort_{c}")
e_lo, e_hi = (effort_cell(pre, c, 80, s2) for s2 in (1.0, 4.0))
st.info(
    f"Bei Auslastung 80 % braucht ein Lauf mit exponentieller Dauer etwa {C.fmt_int(round(customers_needed(e_lo), -3))} Kunden für ±5 %, mit stark streuender Dauer (cs² = 4) "
    f"etwa {C.fmt_int(round(customers_needed(e_hi), -3))}."
)

st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Zwischenankünfte und Dauern sind unabhängig und unkorreliert** | Kingman kennt nur die Varianz; Wellen, Fähren-Pulks und Serien langer Abfertigungen (Autokorrelation) machen die Wartezeit länger, als die Formel sagt. | **[Zeitvariable Ankünfte](https://sebastianhanisch-time-varying-arrivals-demo.streamlit.app/)** |
| **Alle Lkw gleich wichtig** | Eilige Lkw überholen: Für sie wird die Wartezeit kürzer, für die anderen länger; die Mittelwerte hier gelten für alle zusammen. | **[Prioritätsklassen](https://sebastianhanisch-priority-queue-demo.streamlit.app/)** |
| **Ein Gate** | In Netzen verändert jede Station die Streuung des Stroms (Ausgangsstrom ist nicht Poisson); Kingman gilt je Station nur mit angepasstem ca². | **Jackson-Netze** (Folgestück) |
| **Geduldige Lkw, unbegrenzter Warteraum** | Mit Abwanderung oder begrenztem Aufstellplatz ändert sich die Kennzahl: Verlust statt Warten. | **[Erlang A](https://sebastianhanisch-erlang-a-demo.streamlit.app/)** und **[Erlang B](https://sebastianhanisch-erlang-b-demo.streamlit.app/)** |
| **Mehrere Spuren: Allen-Cunneen** | Bei geringer Last ist die Näherung relativ ungenau (der absolute Fehler bleibt winzig); ihr Vorteil liegt bei hoher Last. | kein Folgestück |
| **Nur die mittlere Wartezeit** | Quantile (90 % der Lkw warten höchstens …) brauchen die ganze Verteilung; hier nicht abgebildet. | kein Folgestück |
"""
)
st.caption(
    "Verwandt im Portfolio: [markov-queue-demo](https://sebastianhanisch-markov-queue-demo.streamlit.app/) (Zusatzstück: Phasen-Ketten für Erlang-Dauern ergeben genau die Pollaczek-Khinchine-Wartezeit), [mm1-queue-demo](https://sebastianhanisch-mm1-queue-demo.streamlit.app/) (Stück 1: eine Spur, feste Dauer halbiert die Wartezeit), "
    "[mmc-queue-demo](https://sebastianhanisch-mmc-queue-demo.streamlit.app/) (Stück 3: mehrere Spuren, Erlang C als Basis der Näherung), "
    "[erlang-b-demo](https://sebastianhanisch-erlang-b-demo.streamlit.app/) (Stück 8: der Verlust ist unempfindlich gegen die Streuung, das Warten nicht), "
    "[output-analysis-demo](https://sebastianhanisch-output-analysis-demo.streamlit.app/) (Stück 2: Konfidenzintervalle und Aufwand) und die Hafen-Demo "
    "[truck-appointment-demo](https://sebastianhanisch-truck-appointment-demo.streamlit.app/) (Terminvergabe erzeugt glatte Ankünfte, ca² < 1)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** $G/G/c$: Zwischenankunftszeiten mit Mittel $1/(c\rho)$ und Variationskoeffizient² $c_a^2$, Abfertigungsdauern mit Mittel 1 und $c_s^2$, $c$ Spuren, FIFO, unendlicher Warteraum,
Auslastung $\rho < 1$ je Spur. Wartezeiten in Abfertigungsdauern.

**M/G/1 (exakt).** Pollaczek-Khinchine: $W_q = \dfrac{\lambda\,E[S^2]}{2(1-\rho)} = \dfrac{\rho\,(1 + c_s^2)}{2(1 - \rho)}$.

**G/M/1 (exakt).** $W_q = \dfrac{\sigma}{1 - \sigma}$ mit $\sigma = A^*(1 - \sigma)$, $A^*(s) = E[e^{-sT}]$ die Laplace-Transformierte der Zwischenankunftszeit $T$; bei Poisson-Ankünften
$\sigma = \rho$ (M/M/1).

**Kingman (Näherung, $c = 1$).** $W_q \approx \dfrac{\rho}{1 - \rho}\cdot\dfrac{c_a^2 + c_s^2}{2}$, asymptotisch exakt für $\rho \to 1$ (Schwerverkehr); für $c_a^2 = 1$ gleich Pollaczek-Khinchine.

**Allen-Cunneen (Näherung, $c \ge 1$).** $W_q \approx W_q^{M/M/c}\cdot\dfrac{c_a^2 + c_s^2}{2}$ mit $W_q^{M/M/c} = \dfrac{C(c, c\rho)}{c(1-\rho)}$ (Erlang C).

**Simulation.** Kiefer-Wolfowitz: der Kunde startet bei $\max(\text{Ankunft}, \text{früheste freie Spur})$; für $c = 1$ ist das die Lindley-Rekursion
$W_{n+1} = \max(0, W_n + S_n - A_{n+1})$ (Gegenprobe, derselbe Pfad aus denselben Zufallsströmen).

Implementiert in `mg1_formulas.py` (Formeln, G/M/1), `mg1_simulation.py` (Sampler, Kiefer-Wolfowitz, Lindley), `mg1_evaluation.py` (Live-Lauf, Studienzellen),
`generate_precomputed.py` (vorgerechnete Studie).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html))."
)
