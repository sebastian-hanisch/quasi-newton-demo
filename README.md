# Quasi-Newton (BFGS/L-BFGS) – Krümmung schätzen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-quasi-newton-demo.streamlit.app/)**

Stück 3 der **Nichtlineare-Optimierung-Reihe** der "Konzepte"-Reihe im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Newton (Stück 2) braucht bei jedem Schritt die echte Hesse-Matrix – teuer, und nicht immer leicht
zu bekommen. **BFGS** approximiert die inverse Hesse-Matrix ausschließlich aus
Gradientendifferenzen, ganz ohne sie je zu berechnen. **L-BFGS** speichert nicht einmal die volle
Approximation, nur die letzten $m$ Vektorpaare – Speicherbedarf $O(mn)$ statt $O(n^2)$.

**Einordnung in die Reihe:**

```
Gradientenabstieg (WURZEL)                       [gebaut]
 └─ Newton-Verfahren                             [gebaut]
      └─ Quasi-Newton (BFGS/L-BFGS)               [DIESES STÜCK]
           └─ Lagrange-Multiplikatoren/KKT        [nicht gebaut]
                ├─ Straf-/Barriere-Verfahren      [nicht gebaut]
                └─ SQP                            [nicht gebaut]
                     └─ Innere-Punkte-Verfahren   [nicht gebaut]
 └─ Stochastische Gradientenverfahren             [nicht gebaut, letztes Stück]
```

**Ergebnis in Kürze:** Startet BFGS mit der exakten inversen Hesse-Matrix, reproduziert es Newtons
Schritt bis auf Maschinengenauigkeit ($2{,}8\cdot10^{-16}$). Mit der Standard-Wahl $H_0=I$ und
exakter Liniensuche löst BFGS eine $n$-dimensionale Quadratik in **exakt $n$ Schritten** – aber
nur bis $n\approx20$ in Gleitkomma-Arithmetik; darüber kostet akkumulierter Rundungsfehler einen
moderaten, gemessenen Aufschlag (bei $n=50$ etwa 16 % mehr Schritte). Auf einer Quadratik mit
κ=100 braucht Gradientenabstieg 2337 Schritte, Newton 1, BFGS 20, L-BFGS 94 – BFGS liegt trotz
unbekannter Hesse-Matrix nahe an Newton. **Der Preis von L-BFGS ist real, nicht kosmetisch:**
BFGS bleibt bei jeder Konditionszahl bei 19–23 Schritten, L-BFGS braucht bei κ=500 schon 481 –
und auf der Rosenbrock-Funktion (klassischer Startpunkt) 673 Schritte gegen BFGS' 36.

## Warum dieses Problem

Stück 2 endete mit einer offenen Frage: was, wenn die Hesse-Matrix zu teuer ist oder gar nicht
einfach berechenbar ist? BFGS baut die für Newtons Schritt nötige Information stattdessen
schrittweise aus dem auf, was sowieso berechnet wird: Gradienten.

## Vorab-Hypothesen (vor der Messung notiert, hier geprüft)

| Hypothese | Ergebnis |
|---|---|
| BFGS mit $H_0=A^{-1}$ reproduziert Newtons ersten Schritt exakt | ✅ max. Abweichung $2{,}8\cdot10^{-16}$ |
| BFGS mit $H_0=I$ und exakter Liniensuche löst eine $n$-dim. Quadratik in höchstens $n$ Schritten | ✅ exakt bis $n=20$; danach moderater, gemessener Rundungsfehler-Aufschlag (nie mehr als ~16 % bei den getesteten Größen) |
| BFGS liegt zwischen Gradientenabstieg und Newton, klar näher an Newton | ✅ 2337 (GD) / 1 (Newton) / 20 (BFGS) / 94 (L-BFGS) bei κ=100 |
| Speicherbedarf trennt sich strukturell: BFGS $O(n^2)$, L-BFGS $O(mn)$ | ✅ Faktor 500× bei $n=10.000$ |
| L-BFGS erreicht ungefähr dieselbe Iterationszahl wie BFGS | ⚠️ **Nur bei niedriger Konditionszahl** (κ=5: 21 gegen 23). Bei hoher Konditionszahl braucht L-BFGS deutlich mehr Schritte (κ=500: 481 gegen 19) – der begrenzte Speicher hat einen echten, wachsenden Preis |
| Gradienten-Check gegen finite Differenzen unter $10^{-6}$ | ✅ beide Testfunktionen um $10^{-10}$ |

## Befunde (gemessen, keine Behauptungen)

**Endliche Terminierung auf der Quadratik** (κ=20, $H_0=I$, exakte Liniensuche):

| Dimension $n$ | Iterationen | Innerhalb der Schranke $n$? |
|---|---|---|
| 2 | 2 | Ja |
| 5 | 5 | Ja |
| 10 | 10 | Ja |
| 15 | 15 | Ja |
| 20 | 20 | Ja |
| 25 | 28 | Nein (12 % mehr) |
| 30 | 34 | Nein (13 % mehr) |
| 40 | 46 | Nein (15 % mehr) |
| 50 | 58 | Nein (16 % mehr) |

**Speicherbedarf** (BFGS: volle $n\times n$-Matrix; L-BFGS: $m=10$ Vektorpaare):

| $n$ | BFGS (Zahlen) | L-BFGS (Zahlen) | Faktor |
|---|---|---|---|
| 10 | 100 | 200 | 0,5× |
| 100 | 10.000 | 2.000 | 5× |
| 1.000 | 1.000.000 | 20.000 | 50× |
| 10.000 | 100.000.000 | 200.000 | 500× |

**Vier Verfahren auf derselben Quadratik** (κ=100, $n=10$):

| Verfahren | Iterationen |
|---|---|
| Gradientenabstieg | 2.337 |
| Newton | 1 |
| BFGS | 20 |
| L-BFGS (m=10) | 94 |

**BFGS vs. L-BFGS über die Konditionszahl** ($n=10$):

| κ | BFGS-Iterationen | L-BFGS-Iterationen |
|---|---|---|
| 5 | 23 | 21 |
| 20 | 20 | 50 |
| 100 | 20 | 75 |
| 500 | 19 | 481 |

**Rosenbrock, klassischer Startpunkt $(-1{,}2;\,1{,}0)$:** BFGS 36 Iterationen, L-BFGS
673 Iterationen (beide erreichen $(1,1)$ exakt).

## Modell und Verfahren

- `qn_functions.py` – Quadratik/Rosenbrock (eigenständige Kopie aus den Geschwister-Repos).
- `qn_optimizer.py` – `bfgs_method()` (dichte inverse Hesse-Approximation, Sherman-Morrison-
  Update), `lbfgs_method()` (Zwei-Schleifen-Rekursion), Newton-/Gradientenabstieg-Kopien für die
  Vergleichscharts.
- `qn_evaluation.py` – Reduktions-Check, endliche-Terminierungs-Check, Speicherbedarf-Vergleich,
  Vier-Wege-Vergleich, Konditionszahl-Sweep, Gradienten-Check.
- `qn_visualization.py` – Plotly: 2D-Kontur+Trajektorie, Konvergenzkurve, Vier-Wege-Vergleich,
  Speicherbedarf-vs-$n$, BFGS-vs-L-BFGS-vs-Konditionszahl.

## Was die App zeigt

Funktion (Quadratik/Rosenbrock), Verfahren (BFGS/L-BFGS/Newton/Gradientenabstieg zum Vergleich),
Konditionszahl (nur Quadratik), max. Iterationen und Seed in der Sidebar; Trajektorie und
Konvergenzkurve für die aktuelle Konfiguration; der Vier-Wege-Vergleich als zentraler Befund; ein
"📐"-Abschnitt mit der vollständigen Korrektheits-Kette (Reduktion auf Newton, endliche
Terminierung, Speicherbedarf, Konditionszahl-Sweep, Rosenbrock-Vergleich, Gradienten-Check).

## Was nicht funktioniert hat / Grenzen

**Echte Verfeinerung einer Hypothese (keine volle Widerlegung, aber eine wichtige
Einschränkung):** die Vormessung bestätigte "L-BFGS erreicht ungefähr dieselbe Iterationszahl wie
BFGS" nur bei niedriger Konditionszahl. Bei hoher Konditionszahl (κ=500) braucht L-BFGS 481
Schritte gegen BFGS' 19 – mehr als 25-mal so viele. Der begrenzte Speicher "vergisst"
Krümmungsinformation aus mehr als $m$ Schritten zurück, und genau das kostet bei schwer
konditionierten oder stark gekrümmten (Rosenbrock) Aufgaben spürbar Zeit. Interessanterweise half
eine Erhöhung von $m$ (3 bis 20 getestet) bei Rosenbrock kaum – die Zahl blieb bei
ca. 670–675 Iterationen, unabhängig von $m$ in diesem Bereich.

**Grenzen:** die Liniensuche ist bewusst einfach gehalten (Backtracking/Armijo plus ein simpler
Krümmungs-Sicherheitscheck $y^\top s>0$) statt einer vollen Wolfe-Bedingungs-Suche – ausreichend
für die hier gezeigten Effekte, aber nicht production-grade. Nur unrestringierte Minimierung.

## Tests

32 Tests, `python -m pytest tests/ -v` (Laufzeit lokal ~2 Sekunden):
- `test_functions.py` – Testfunktionen, Gradienten gegen finite Differenzen.
- `test_optimizer.py` – Reduktion auf Newton, endliche Terminierung, BFGS/L-BFGS auf Rosenbrock.
- `test_evaluation.py` – Sweep-Funktionen mit billigen Parametern.
- `test_claims.py` – jede Zahl oben nachgerechnet, mit Toleranzband (Modul-Fixtures für die
  teureren Sweeps).
- `test_presets.py`, `test_app.py` – Presets, Funktions-/Verfahrenswechsel, Footer.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `qn_constants.py` | Regler-Grenzen, Presets |
| `qn_functions.py` | Testfunktionen (Quadratik, Rosenbrock) |
| `qn_optimizer.py` | BFGS, L-BFGS, Newton-/Gradientenabstieg-Kopien |
| `qn_evaluation.py` | Sweeps, Korrektheits-Kette, Gradienten-Check |
| `qn_visualization.py` | Plotly-Plots |
| `qn_presets.py` | Permalink-Sync, Presets |
| `tests/` | pytest-Suite |

## Bewusst nicht umgesetzt

Keine volle Wolfe-Bedingungs-Liniensuche (siehe Grenzen oben). Kein Vergleich gegen SciPy – die
Korrektheit wird gegen das bekannte exakte Optimum und Newtons eigenen Schritt geprüft, eine
stärkere Garantie als ein Solver-Kreuzvergleich. Kein Regler für die Rosenbrock-Startposition –
bewusst fest auf den klassischen Literatur-Startpunkt gesetzt, damit App und README dieselbe Zahl
zeigen.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements-dev.txt
streamlit run app.py
```

## Literatur

- Broyden, C. G. (1970); Fletcher, R. (1970); Goldfarb, D. (1970); Shanno, D. F. (1970) – die
  vier unabhängigen BFGS-Originalarbeiten.
- Liu, D. C. & Nocedal, J. (1989). *On the limited memory BFGS method for large scale
  optimization.* Mathematical Programming, 45, 503–528.
- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization* (2. Aufl.). Springer.
