"""Plotly-Abbildungen der M/G/1-Kingman-Demo: Pollaczek-Khinchine-Kurven, Vergleich von Simulation und Formeln, Fehler der Näherung (Heatmap und Verlauf über die Auslastung),
Aufwand für eine Genauigkeit. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import plotly.graph_objects as go

import mg1_constants as C
import mg1_evaluation as E
import mg1_formulas as F

CS2_COLORS = {0.0: "#72b7b2", 0.25: "#54a24b", 1.0: "#4c78a8", 4.0: "#e45756"}
CA2_COLORS = {0.0: "#72b7b2", 1.0: "#4c78a8", 4.0: "#e45756"}
SIM_COLOR = "#f58518"
APPROX_COLOR = "#b279a2"
EXACT_COLOR = "#4c78a8"
BASE_COLOR = "#9d9d9d"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.25, top=10):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=top, b=10), legend=dict(orientation="h", y=legend_y),
                      plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def ca2_short(ca2):
    return {0.0: "glatt", 1.0: "Poisson", 4.0: "stoßweise"}.get(ca2, f"{ca2:g}")


def cs2_short(cs2):
    return {0.0: "fest", 0.25: "Erlang-4", 1.0: "exponentiell", 4.0: "streuend"}.get(cs2, f"{cs2:g}")


def build_pk_chart(rho_pct, cs2, wait_sim=None):
    """Wartezeit in Minuten über die Auslastung (M/G/1, Pollaczek-Khinchine, exakt) für vier Streuungen der Dauer; der Punkt ist der Simulationslauf der Einstellung."""
    rhos = [r / 100 for r in range(30, 96)]
    fig = go.Figure()
    for x in C.CS2_OPTIONS:
        fig.add_trace(go.Scatter(x=[100 * r for r in rhos], y=[F.to_minutes(F.pk_wait(r, x)) for r in rhos], mode="lines",
                                 line=dict(color=CS2_COLORS[x], width=3 if x == cs2 else 1.8), name=f"Dauer {cs2_short(x)} (cs² = {x:g})"))
    if wait_sim is not None:
        fig.add_trace(go.Scatter(x=[rho_pct], y=[F.to_minutes(wait_sim)], mode="markers", marker=dict(color=SIM_COLOR, size=11, symbol="diamond"),
                                 name="Simulation (gewählt)"))
    fig.update_xaxes(title_text="Auslastung der Spur", ticksuffix=" %")
    fig.update_yaxes(title_text="mittlere Wartezeit in Minuten (logarithmisch)", type="log")
    return _base(fig, 340, legend_y=-0.3)


def build_compare_chart(report, c, ca2, cs2):
    """Wartezeit in Minuten für die gewählte Einstellung: M/M/c (Bezug), Näherung (Kingman bzw. Allen-Cunneen), exakter Wert (wo vorhanden), Simulation."""
    labels, values, colors = [f"M/M/{c} (Bezug)"], [F.to_minutes(report["mm"])], [BASE_COLOR]
    labels.append("Näherung (Kingman)" if c == 1 else "Näherung (Allen-Cunneen)")
    values.append(F.to_minutes(report["approx"]))
    colors.append(APPROX_COLOR)
    if report["exact"] is not None:
        labels.append("exakt")
        values.append(F.to_minutes(report["exact"]))
        colors.append(EXACT_COLOR)
    labels.append("Simulation")
    values.append(F.to_minutes(report["wait_sim"]))
    colors.append(SIM_COLOR)
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=[f"{v:.2f}" for v in values], textposition="outside"))
    fig.update_yaxes(title_text="mittlere Wartezeit in Minuten", rangemode="tozero")
    return _base(fig, 340, top=20)


def build_error_heatmap(pre, c, rho_pct):
    """Fehler der Näherung gegen die Simulation (Näherung/Simulation − 1, in %) über Streuung der Ankünfte (Zeilen) und der Dauer (Spalten); leere Zelle: keine Wartezeit."""
    z, text = [], []
    for ca2 in C.STUDY_CA2:
        row, trow = [], []
        for cs2 in C.STUDY_CS2:
            err = E.approx_error(E.study_cell(pre, c, rho_pct, ca2, cs2))
            row.append(None if err is None else 100 * err)
            trow.append("–" if err is None else C.fmt_signed_pct(err))
        z.append(row)
        text.append(trow)
    fig = go.Figure(go.Heatmap(z=z, x=[cs2_short(x) for x in C.STUDY_CS2], y=[ca2_short(x) for x in C.STUDY_CA2], text=text, texttemplate="%{text}",
                               colorscale="RdBu", reversescale=True, zmid=0, zmin=-100, zmax=100, showscale=False))
    fig.update_xaxes(title_text="Streuung der Abfertigungsdauer")
    fig.update_yaxes(title_text="Streuung der Ankünfte")
    return _base(fig, 300, top=10)


def build_heavy_traffic_chart(pre, c, cs2):
    """Fehler der Näherung (Näherung/Simulation − 1) über die Auslastung für drei Streuungen der Ankünfte: er schrumpft in starkem Verkehr."""
    fig = go.Figure()
    for ca2 in C.STUDY_CA2:
        xs, ys = [], []
        for rho_pct in C.STUDY_RHO_PCT:
            err = E.approx_error(E.study_cell(pre, c, rho_pct, ca2, cs2))
            if err is not None:
                xs.append(rho_pct)
                ys.append(100 * err)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=CA2_COLORS[ca2], width=2.5), name=f"Ankünfte {ca2_short(ca2)}"))
    fig.add_hline(y=0, line=dict(color=BASE_COLOR, dash="dash"))
    fig.update_xaxes(title_text="Auslastung der Spur", tickvals=list(C.STUDY_RHO_PCT), ticksuffix=" %")
    fig.update_yaxes(title_text="Näherung gegen Simulation", ticksuffix=" %")
    return _base(fig, 300)


def build_effort_chart(pre, c):
    """Kunden für eine relative Streuung von 5 % eines Laufs (logarithmisch) über die Streuung der Dauer, für drei Auslastungen (Poisson-Ankünfte, je 48 Läufe à 100 000 Kunden)."""
    fig = go.Figure()
    colors = {50: "#9ecae9", 80: "#4c78a8", 95: "#1f3d63"}
    for rho_pct in C.STUDY_RHO_PCT:
        xs, ys = [], []
        for cs2 in C.STUDY_CS2:
            need = E.customers_needed(E.effort_cell(pre, c, rho_pct, cs2))
            if need == need:
                xs.append(cs2)
                ys.append(need)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=colors[rho_pct], width=2.5), name=f"Auslastung {rho_pct} %"))
    fig.update_xaxes(title_text="Streuung der Abfertigungsdauer cs²", tickvals=list(C.STUDY_CS2))
    fig.update_yaxes(title_text="Kunden für ±5 % (logarithmisch)", type="log")
    return _base(fig, 300)
