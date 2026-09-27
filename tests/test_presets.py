import qn_constants as C
import qn_evaluation as ev


def test_all_presets_have_valid_settings():
    for key, preset in C.PRESETS.items():
        assert preset["func"] in C.FUNCTIONS
        assert preset["optimizer"] in C.OPTIMIZERS


def test_preset_quadratik_bfgs_converges_fast():
    p = C.PRESETS["quadratik_bfgs"]
    settings = ev.Settings(func="quadratic", optimizer="bfgs", max_iter=p["max_iter"],
                           seed=p["seed"], condition_number=p["kappa"])
    out = ev.analyse(settings)
    assert out["result"].n_iter <= 2


def test_preset_rosenbrock_bfgs_converges():
    p = C.PRESETS["rosenbrock_bfgs"]
    settings = ev.Settings(func="rosenbrock", optimizer="bfgs", max_iter=p["max_iter"],
                           seed=p["seed"])
    out = ev.analyse(settings)
    assert out["result"].converged


def test_preset_rosenbrock_lbfgs_converges_with_more_iterations():
    p_bfgs = C.PRESETS["rosenbrock_bfgs"]
    p_lbfgs = C.PRESETS["rosenbrock_lbfgs"]
    out_bfgs = ev.analyse(ev.Settings(func="rosenbrock", optimizer="bfgs",
                                      max_iter=p_bfgs["max_iter"], seed=p_bfgs["seed"]))
    out_lbfgs = ev.analyse(ev.Settings(func="rosenbrock", optimizer="lbfgs",
                                       max_iter=p_lbfgs["max_iter"], seed=p_lbfgs["seed"]))
    assert out_lbfgs["result"].converged
    assert out_lbfgs["result"].n_iter > out_bfgs["result"].n_iter
