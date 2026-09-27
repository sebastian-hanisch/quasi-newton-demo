"""Reine Plotly-Figure-Builder, keine Streamlit-Aufrufe. Achsen fest uebergeben (siehe
feedback_plotly_fixedrange_convention/feedback_plotly_scaleanchor_explicit_range). Balken-x-Achsen
nutzen echte Zahlenwerte (feedback_plotly_bar_numeric_labels_vline_mismatch). Keine literalen "|"
in Markdown-Tabellenzellen (feedback_markdown_table_literal_pipe_breaks_columns)."""
import numpy as np
import plotly.graph_objects as go

COLOR_PATH = "#1f77b4"
COLOR_START = "#d62728"
COLOR_OPT = "#2ca02c"
COLOR_GD = "#d62728"
COLOR_NEWTON = "#2ca02c"
COLOR_BFGS = "#1f77b4"
COLOR_LBFGS = "#9467bd"


def build_trajectory_figure_2d(f, trajectory, x_star, x_range, y_range, title=""):
    xs = np.linspace(x_range[0], x_range[1], 120)
    ys = np.linspace(y_range[0], y_range[1], 120)
    Z = np.zeros((len(ys), len(xs)))
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            Z[i, j] = f(np.array([xv, yv]))
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=np.log1p(np.maximum(Z, 0)), showscale=False, colorscale="Blues",
        contours=dict(coloring="fill"), opacity=0.75,
    ))
    fig.add_trace(go.Scatter(
        x=trajectory[:, 0], y=trajectory[:, 1], mode="lines+markers", name="Pfad",
        line=dict(color=COLOR_PATH, width=2), marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=[trajectory[0, 0]], y=[trajectory[0, 1]], mode="markers", name="Start",
        marker=dict(color=COLOR_START, size=12, symbol="x"),
    ))
    fig.add_trace(go.Scatter(
        x=[x_star[0]], y=[x_star[1]], mode="markers", name="Optimum",
        marker=dict(color=COLOR_OPT, size=13, symbol="star"),
    ))
    fig.update_layout(
        title=title, xaxis=dict(range=list(x_range), fixedrange=True, title="x₁"),
        yaxis=dict(range=list(y_range), fixedrange=True, title="x₂"),
        showlegend=True, height=420, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_convergence_figure(fvals, f_star=0.0, title="Konvergenz: f(xₖ) - f*"):
    gap = np.maximum(np.array(fvals) - f_star, 1e-300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(gap))), y=gap, mode="lines+markers", name="f(xₖ) - f*",
        line=dict(color=COLOR_PATH, width=2),
    ))
    fig.update_layout(
        title=title, xaxis=dict(title="Iteration k", fixedrange=True),
        yaxis=dict(title="f(xₖ) - f*", fixedrange=True, type="log"),
        showlegend=False, height=320, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_four_way_figure(comparison, f_star=0.0, title="Gradientenabstieg vs. Newton vs. BFGS vs. L-BFGS"):
    fig = go.Figure()
    for key, name, color in (("gd", "Gradientenabstieg", COLOR_GD), ("newton", "Newton", COLOR_NEWTON),
                             ("bfgs", "BFGS", COLOR_BFGS), ("lbfgs", "L-BFGS", COLOR_LBFGS)):
        result = comparison[key]
        gap = np.maximum(np.array(result.fvals) - f_star, 1e-300)
        fig.add_trace(go.Scatter(x=list(range(len(gap))), y=gap, mode="lines", name=name,
                                 line=dict(color=color, width=2)))
    fig.update_layout(
        title=title, xaxis=dict(title="Iteration k", fixedrange=True, type="log"),
        yaxis=dict(title="f(xₖ) - f*", fixedrange=True, type="log"),
        height=380, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_finite_termination_figure(rows, title="Endliche Terminierung: Iterationen vs. Dimension n"):
    ns = [r["n"] for r in rows]
    iters = [r["n_iter"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=ns, mode="lines", name="Theoretische Schranke (=n)",
                             line=dict(color="#d62728", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=ns, y=iters, mode="lines+markers", name="Gemessene Iterationen",
                             line=dict(color=COLOR_BFGS, width=2)))
    fig.update_layout(
        title=title, xaxis=dict(title="Dimension n", fixedrange=True),
        yaxis=dict(title="Iterationen bis Konvergenz", fixedrange=True),
        height=340, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_memory_footprint_figure(rows, title="Speicherbedarf: BFGS (n²) vs. L-BFGS (mn)"):
    ns = [r["n"] for r in rows]
    bfgs = [r["bfgs_numbers"] for r in rows]
    lbfgs = [r["lbfgs_numbers"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=bfgs, mode="lines+markers", name="BFGS (n²)",
                             line=dict(color=COLOR_BFGS, width=2)))
    fig.add_trace(go.Scatter(x=ns, y=lbfgs, mode="lines+markers", name="L-BFGS (m·n, m=10)",
                             line=dict(color=COLOR_LBFGS, width=2)))
    fig.update_layout(
        title=title, xaxis=dict(title="Dimension n", fixedrange=True, type="log"),
        yaxis=dict(title="Gespeicherte Zahlen", fixedrange=True, type="log"),
        height=340, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_condition_sweep_figure(rows, title="BFGS vs. L-BFGS: Iterationen vs. Konditionszahl"):
    kappas = [r["kappa"] for r in rows]
    bfgs_iters = [r["bfgs_iters"] for r in rows]
    lbfgs_iters = [r["lbfgs_iters"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=kappas, y=bfgs_iters, mode="lines+markers", name="BFGS",
                             line=dict(color=COLOR_BFGS, width=2)))
    fig.add_trace(go.Scatter(x=kappas, y=lbfgs_iters, mode="lines+markers", name="L-BFGS (m=10)",
                             line=dict(color=COLOR_LBFGS, width=2)))
    fig.update_layout(
        title=title, xaxis=dict(title="Konditionszahl κ", fixedrange=True, type="log"),
        yaxis=dict(title="Iterationen bis Konvergenz", fixedrange=True, type="log"),
        height=340, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig
