"""Quasi-Newton (BFGS/L-BFGS) — Krümmung schätzen

Sebastian Hanisch - Operations Research und Machine Learning

Stück 3 der "Nichtlineare Optimierung"-Reihe der "Konzepte"-Reihe:
Gradientenabstieg -> Newton-Verfahren -> Quasi-Newton -> Lagrange/KKT ->
{Straf-/Barriere-Verfahren, SQP -> Innere-Punkte-Verfahren} + Stochastische Gradientenverfahren.
Stück 2 nutzte die echte Hesse-Matrix. Quasi-Newton (BFGS) approximiert sie ausschließlich aus
Gradientendifferenzen, ganz ohne sie je zu berechnen. L-BFGS speichert nicht einmal die volle
Approximation, sondern nur die letzten m Vektorpaare.

Lauffähig mit: streamlit run app.py
"""
import numpy as np
import streamlit as st

import qn_constants as C
import qn_evaluation as ev
import qn_functions as fn
import qn_presets as pr
import qn_visualization as viz

st.set_page_config(page_title="Quasi-Newton", layout="wide")

FUNC_LABELS = {C.FUNC_QUADRATIC: "Quadratik (κ regelbar)", C.FUNC_ROSENBROCK: "Rosenbrock-Funktion"}
OPT_LABELS = {C.OPT_BFGS: "BFGS", C.OPT_LBFGS: "L-BFGS", C.OPT_NEWTON: "Newton (Vergleich)",
              C.OPT_GD: "Gradientenabstieg (Vergleich)"}


@st.cache_data(show_spinner=False)
def _run(func, optimizer, kappa, max_iter, seed):
    import qn_optimizer as opt
    if func == C.FUNC_QUADRATIC:
        A = fn.random_spd_matrix(C.DIM, kappa, seed)
        f, grad, hess = fn.quadratic(A)
        rng = np.random.default_rng(seed + 1)
        x_start = rng.normal(size=C.DIM)
        x_star = np.zeros(C.DIM)
    else:
        f, grad, hess = fn.rosenbrock()
        x_start = np.array(C.ROSENBROCK_X0)
        x_star = np.array([1.0, 1.0])

    if optimizer == C.OPT_BFGS:
        result = opt.bfgs_method(f, grad, x_start, max_iter=max_iter)
    elif optimizer == C.OPT_LBFGS:
        result = opt.lbfgs_method(f, grad, x_start, m=10, max_iter=max_iter)
    elif optimizer == C.OPT_NEWTON:
        result = opt.newton_method(f, grad, hess, x_start, max_iter=max_iter)
    else:
        lam_max = float(np.linalg.eigvalsh(fn.random_spd_matrix(C.DIM, kappa, seed)).max()) \
            if func == C.FUNC_QUADRATIC else 400.0
        result = opt.gradient_descent(f, grad, x_start, eta=1.0 / lam_max, max_iter=max_iter)
    return {"trajectory": result.trajectory, "fvals": result.fvals, "converged": result.converged,
            "n_iter": result.n_iter, "x_star": x_star}


def _rebuild_function(func, kappa, seed):
    if func == C.FUNC_QUADRATIC:
        A = fn.random_spd_matrix(C.DIM, kappa, seed)
        f, grad, hess = fn.quadratic(A)
        return f
    f, grad, hess = fn.rosenbrock()
    return f


@st.cache_data(show_spinner=False)
def _reduction_check():
    return ev.reduction_to_newton_check()


@st.cache_data(show_spinner=False)
def _finite_termination():
    return ev.finite_termination_check()


@st.cache_data(show_spinner=False)
def _memory_footprint():
    return ev.memory_footprint_comparison()


@st.cache_data(show_spinner=False)
def _four_way():
    out = ev.four_way_comparison()
    return {k: {"fvals": v.fvals, "n_iter": v.n_iter} for k, v in out.items()}


class _R:
    pass


@st.cache_data(show_spinner=False)
def _condition_sweep():
    return ev.lbfgs_vs_bfgs_condition_sweep()


@st.cache_data(show_spinner=False)
def _rosenbrock_comparison():
    out = ev.rosenbrock_bfgs_vs_lbfgs()
    return {"bfgs_iters": out["bfgs"].n_iter, "bfgs_final": out["bfgs"].fvals[-1],
            "lbfgs_iters": out["lbfgs"].n_iter, "lbfgs_final": out["lbfgs"].fvals[-1]}


@st.cache_data(show_spinner=False)
def _gradient_check():
    return ev.gradient_check()


st.title("🧭 Quasi-Newton (BFGS/L-BFGS) — Krümmung schätzen")
st.markdown(
    "Newton (Stück 2) braucht bei jedem Schritt die echte Hesse-Matrix — teuer, und nicht immer "
    "leicht zu bekommen. **BFGS** baut stattdessen eine Approximation der inversen Hesse-Matrix "
    "**ausschließlich aus Gradientendifferenzen** auf, ganz ohne sie je zu berechnen. **L-BFGS** "
    "geht noch weiter: es speichert nicht einmal die volle Approximation, nur die letzten $m$ "
    "Vektorpaare — Speicherbedarf $O(mn)$ statt $O(n^2)$."
)
st.caption(
    "Stück 3 der 'Nichtlineare Optimierung'-Reihe. Folgestücke (alle gebaut): "
    "Lagrange/KKT, Straf-/Barriere-Verfahren, SQP, Innere-Punkte-Verfahren, Stochastische "
    "Gradientenverfahren."
)

with st.expander("So funktioniert BFGS/L-BFGS", expanded=True):
    st.markdown(
        "1. Wie Newton: $x_{k+1}=x_k-H_k\\nabla f(x_k)$ — aber $H_k$ ist nur eine Schätzung der "
        "inversen Hesse-Matrix, keine echte.\n"
        "2. Nach jedem Schritt wird $H_k$ aus der Differenz der Punkte "
        "$s_k=x_{k+1}-x_k$ und der Gradienten $y_k=\\nabla f(x_{k+1})-\\nabla f(x_k)$ "
        "aktualisiert (BFGS-Formel) — keine einzige Auswertung der echten Hesse-Matrix nötig.\n"
        "3. **L-BFGS** speichert dafür nur die letzten $m$ Paare $(s_k,y_k)$ und rekonstruiert "
        "die Wirkung von $H_k$ bei Bedarf (Zwei-Schleifen-Rekursion) — nie die volle Matrix."
    )

st.caption("🎯 Schnellstart – ein Klick lädt ein durchgerechnetes Beispiel:")
preset_cols = st.columns(len(C.PRESETS))
for col, (key, preset) in zip(preset_cols, C.PRESETS.items()):
    with col:
        st.button(preset["label"], help=preset["help"], on_click=pr.apply_preset, args=(key,),
                   use_container_width=True)

st.caption("🔗 Die Adresszeile speichert deine Einstellungen als Permalink.")

pr.load_permalink_settings()
pr.init_session_state_defaults()
ss = st.session_state

with st.sidebar:
    st.header("⚙️ Einstellungen")
    func = st.radio("Funktion", C.FUNCTIONS, format_func=lambda f: FUNC_LABELS[f],
                    key="widget_func", index=C.FUNCTIONS.index(ss["func"]),
                    on_change=pr.store_from_widget, args=("func",))
    ss["func"] = func

    optimizer = st.radio("Verfahren", C.OPTIMIZERS, format_func=lambda o: OPT_LABELS[o],
                         key="widget_optimizer", index=C.OPTIMIZERS.index(ss["optimizer"]),
                         on_change=pr.store_from_widget, args=("optimizer",))
    ss["optimizer"] = optimizer

    if func == C.FUNC_QUADRATIC:
        kappa = st.slider("Konditionszahl κ", C.KAPPA_MIN, C.KAPPA_MAX, ss["kappa"], step=0.5,
                          key="widget_kappa", on_change=pr.store_from_widget, args=("kappa",))
        ss["kappa"] = kappa
    else:
        kappa = ss["kappa"]
        st.caption(f"Startpunkt fest bei {C.ROSENBROCK_X0} (klassischer Literatur-Startwert).")

    max_iter = st.slider("Max. Iterationen", C.MAX_ITER_MIN, C.MAX_ITER_MAX, ss["max_iter"],
                         key="widget_max_iter", on_change=pr.store_from_widget, args=("max_iter",))
    ss["max_iter"] = max_iter
    seed = st.number_input("Seed", value=ss["seed"], step=1, key="widget_seed",
                           on_change=pr.store_from_widget, args=("seed",))
    ss["seed"] = seed
    st.button("🎲 Zufälliger Seed", on_click=pr.randomize_seed)

pr.sync_query_params(dict(func=func, optimizer=optimizer, kappa=kappa, max_iter=max_iter,
                         seed=seed))

out = _run(func, optimizer, kappa, int(max_iter), int(seed))
f = _rebuild_function(func, kappa, int(seed))

st.markdown("---")
st.subheader("🎯 Der Weg zum Minimum")
col_left, col_right = st.columns([3, 2])
with col_left:
    if func == C.FUNC_QUADRATIC:
        pad = max(1.0, float(np.abs(out["trajectory"]).max()) * 1.2)
        x_range = y_range = (-pad, pad)
    else:
        x_range, y_range = (-2.0, 2.0), (-1.0, 3.0)
    fig_traj = viz.build_trajectory_figure_2d(f, out["trajectory"], out["x_star"], x_range,
                                              y_range,
                                              title=f"Pfad ({FUNC_LABELS[func]}, {OPT_LABELS[optimizer]})")
    st.plotly_chart(fig_traj, key=f"traj_{func}_{optimizer}_{kappa}_{max_iter}_{seed}",
                    use_container_width=True)
with col_right:
    fig_conv = viz.build_convergence_figure(out["fvals"], 0.0)
    st.plotly_chart(fig_conv, key=f"conv_{func}_{optimizer}_{kappa}_{max_iter}_{seed}",
                    use_container_width=True)

st.subheader("🎯 Was am Ende steht")
m1, m2, m3 = st.columns(3)
m1.metric("Konvergiert?", "Ja" if out["converged"] else "Nein (Max. Iterationen erreicht)")
m2.metric("Iterationen", f"{out['n_iter']}")
m3.metric("f(x) am Ende", f"{out['fvals'][-1]:.2e}")

st.markdown("---")
st.subheader("🎯 Die zentrale Messung: vier Verfahren auf derselben Quadratik (κ=100)")
four_way = _four_way()


def _to_result(d):
    r = _R()
    r.fvals = d["fvals"]
    return r


comparison = {k: _to_result(v) for k, v in four_way.items()}
st.plotly_chart(viz.build_four_way_figure(comparison), key="four_way_chart",
                use_container_width=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Gradientenabstieg", f"{four_way['gd']['n_iter']} Iter.")
c2.metric("Newton", f"{four_way['newton']['n_iter']} Iter.")
c3.metric("BFGS", f"{four_way['bfgs']['n_iter']} Iter.")
c4.metric("L-BFGS", f"{four_way['lbfgs']['n_iter']} Iter.")
st.caption(
    "BFGS braucht keine Hesse-Matrix und liegt trotzdem nahe an Newtons Geschwindigkeit — weit "
    "vor Gradientenabstieg. L-BFGS ist hier langsamer als volles BFGS, aber immer noch weit vor "
    "Gradientenabstieg."
)

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    "| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |\n"
    "|---|---|---|\n"
    "| Liniensuche erfüllt die Krümmungsbedingung | Update wird übersprungen, kein Fortschritt "
    "in diesem Schritt | Praxis-Sicherung (hier bereits eingebaut) |\n"
    "| Begrenzter Speicher (L-BFGS) reicht für die Krümmung des Problems | Braucht bei "
    "schlecht konditionierten/stark gekrümmten Aufgaben deutlich mehr Schritte (siehe 📐) | "
    "Größeres m, oder volles BFGS |\n"
    "| Keine Nebenbedingungen | Reine unrestringierte Minimierung | Lagrange/KKT (Stück 4) |\n"
)

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**BFGS-Update:** $H_{k+1}=(I-\rho_k s_k y_k^\top)H_k(I-\rho_k y_k s_k^\top)+\rho_k s_k s_k^\top$,
$\rho_k=1/(y_k^\top s_k)$.

**Reduktion auf Newton:** startet BFGS mit $H_0=A^{-1}$ (der exakten inversen Hesse-Matrix einer
Quadratik), ist der erste Schritt algebraisch identisch zu Newtons Schritt aus Stück 2.
"""
    )
    err = _reduction_check()
    st.metric("Reduktion auf Newton: max. Abweichung der Trajektorie", f"{err:.2e}")

    st.markdown(
        "**Endliche Terminierung:** BFGS mit $H_0=I$ und exakter Liniensuche sollte eine "
        "$n$-dimensionale Quadratik in höchstens $n$ Schritten lösen (klassisches Ergebnis)."
    )
    ft_rows = _finite_termination()
    st.plotly_chart(viz.build_finite_termination_figure(ft_rows), key="finite_termination_chart",
                    use_container_width=True)
    st.caption(
        "Hält exakt bis n≈20 in Gleitkomma-Arithmetik; darüber kostet akkumulierter "
        "Rundungsfehler einen moderaten Aufschlag (bei n=50 etwa 16% mehr Schritte als die "
        "theoretische Schranke), kein Zusammenbruch."
    )

    st.markdown("**Speicherbedarf:** BFGS speichert eine volle $n\\times n$-Matrix, L-BFGS nur "
               "$m$ Vektorpaare der Länge $n$ (hier $m=10$):")
    mem_rows = _memory_footprint()
    st.plotly_chart(viz.build_memory_footprint_figure(mem_rows), key="memory_chart",
                    use_container_width=True)

    st.markdown("**Ehrlicher Befund:** BFGS bleibt bei jeder Konditionszahl bei einer ähnlich "
               "niedrigen Iterationszahl, L-BFGS braucht mit wachsender Konditionszahl deutlich "
               "mehr Schritte — der begrenzte Speicher hat einen echten Preis:")
    cs_rows = _condition_sweep()
    st.plotly_chart(viz.build_condition_sweep_figure(cs_rows), key="condition_sweep_chart",
                    use_container_width=True)

    rb = _rosenbrock_comparison()
    r1, r2 = st.columns(2)
    r1.metric("Rosenbrock, klassischer Startpunkt: BFGS", f"{rb['bfgs_iters']} Iterationen")
    r2.metric("Rosenbrock, klassischer Startpunkt: L-BFGS", f"{rb['lbfgs_iters']} Iterationen")

    grad_err = _gradient_check()
    g1, g2 = st.columns(2)
    g1.metric("Gradienten-Check Quadratik", f"{grad_err['quadratic_grad_max_rel_err']:.1e}")
    g2.metric("Gradienten-Check Rosenbrock", f"{grad_err['rosenbrock_grad_max_rel_err']:.1e}")

    st.markdown(
        "**Literatur:** Broyden, C. G.; Fletcher, R.; Goldfarb, D.; Shanno, D. F. (je 1970). "
        "Liu, D. C. & Nocedal, J. (1989). *On the limited memory BFGS method for large scale "
        "optimization.* Mathematical Programming, 45, 503–528. Nocedal, J. & Wright, S. J. "
        "(2006). *Numerical Optimization* (2. Aufl.). Springer."
    )
    st.caption(
        "Implementiert in `qn_functions.py` (Testfunktionen), `qn_optimizer.py` (BFGS, L-BFGS, "
        "Newton-/Gradientenabstieg-Kopien), `qn_evaluation.py` (Korrektheits-Kette, Sweeps), "
        "`qn_visualization.py` (Plots)."
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Nichtlineare Optimierung: acht Stücke, zwei Äste](https://sebastianhanisch.net/konzepte-nichtlineare-optimierung.html)."
)
