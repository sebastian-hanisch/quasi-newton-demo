from streamlit.testing.v1 import AppTest

_APP_TIMEOUT = 60


def _fresh():
    at = AppTest.from_file("../app.py", default_timeout=_APP_TIMEOUT)
    at.run()
    return at


def test_app_runs_without_exception():
    at = _fresh()
    assert not at.exception


def test_footer_is_present():
    at = _fresh()
    captions = [c.value for c in at.caption]
    assert any("Sebastian Hanisch" in c and "Kontakt aufnehmen" in c for c in captions)


def test_preset_quadratik_bfgs():
    at = _fresh()
    btn = [b for b in at.button if b.label == "Quadratik — BFGS terminiert exakt"][0]
    btn.click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Konvergiert?"] == "Ja"


def test_preset_rosenbrock_lbfgs():
    at = _fresh()
    btn = [b for b in at.button if b.label == "Rosenbrock — L-BFGS"][0]
    btn.click().run()
    assert not at.exception


def test_switching_functions_and_optimizers_does_not_crash():
    at = _fresh()
    radio = [r for r in at.radio if r.label == "Funktion"][0]
    radio.set_value("rosenbrock").run()
    assert not at.exception
    opt_radio = [r for r in at.radio if r.label == "Verfahren"][0]
    opt_radio.set_value("lbfgs").run()
    assert not at.exception
