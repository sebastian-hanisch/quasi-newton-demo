"""Regler-Grenzen, feste Annahmen, gemessene Werte und Presets."""

DEFAULT_SEED = 0

FUNC_QUADRATIC = "quadratik"
FUNC_ROSENBROCK = "rosenbrock"
FUNCTIONS = (FUNC_QUADRATIC, FUNC_ROSENBROCK)

OPT_BFGS = "bfgs"
OPT_LBFGS = "lbfgs"
OPT_NEWTON = "newton"
OPT_GD = "gd"
OPTIMIZERS = (OPT_BFGS, OPT_LBFGS, OPT_NEWTON, OPT_GD)

DIM = 2  # feste Dimension fuer die interaktive Quadratik (2D-Kontur/Trajektorie sichtbar)

KAPPA_MIN, KAPPA_MAX, KAPPA_DEFAULT = 1.5, 500.0, 20.0
MAX_ITER_MIN, MAX_ITER_MAX, MAX_ITER_DEFAULT = 1, 1000, 50

# Rosenbrock nutzt in dieser App immer den klassischen, in der Literatur zitierten Startpunkt
# (-1.2, 1.0) - kein Regler, damit die BFGS-vs-L-BFGS-Messung im Kopf und in der App
# uebereinstimmen (siehe README: bei einem Diagonal-Startpunkt ist der Unterschied viel kleiner).
ROSENBROCK_X0 = (-1.2, 1.0)

PRESETS = {
    "quadratik_bfgs": dict(
        label="Quadratik — BFGS terminiert exakt",
        func=FUNC_QUADRATIC, optimizer=OPT_BFGS, kappa=100.0, max_iter=30, seed=0,
        help="BFGS mit exakter Liniensuche löst eine 2D-Quadratik in höchstens 2 Schritten.",
    ),
    "rosenbrock_bfgs": dict(
        label="Rosenbrock — BFGS",
        func=FUNC_ROSENBROCK, optimizer=OPT_BFGS, kappa=KAPPA_DEFAULT, max_iter=50, seed=0,
        help="BFGS erreicht das Optimum in wenigen Dutzend Schritten, ohne je die Hesse-Matrix "
             "zu berechnen.",
    ),
    "rosenbrock_lbfgs": dict(
        label="Rosenbrock — L-BFGS",
        func=FUNC_ROSENBROCK, optimizer=OPT_LBFGS, kappa=KAPPA_DEFAULT, max_iter=1000, seed=0,
        help="L-BFGS braucht auf demselben Startpunkt deutlich mehr Schritte als BFGS — der "
             "begrenzte Speicher hat einen echten Preis.",
    ),
}
