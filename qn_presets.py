"""Permalink-Sync (Query-Parameter <-> Session-State) und Presets."""
from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st

import qn_constants as C


@dataclass(frozen=True)
class SettingSpec:
    key: str
    param: str
    default: Any
    cast: Callable[[str], Any]
    bounds: tuple | None = None


SETTING_SPECS = [
    SettingSpec("func", "f", C.FUNC_QUADRATIC, str),
    SettingSpec("optimizer", "opt", C.OPT_BFGS, str),
    SettingSpec("kappa", "kappa", C.KAPPA_DEFAULT, float, (C.KAPPA_MIN, C.KAPPA_MAX)),
    SettingSpec("max_iter", "iter", C.MAX_ITER_DEFAULT, int, (C.MAX_ITER_MIN, C.MAX_ITER_MAX)),
    SettingSpec("seed", "seed", C.DEFAULT_SEED, int),
]


def init_session_state_defaults() -> None:
    for spec in SETTING_SPECS:
        if spec.key not in st.session_state:
            st.session_state[spec.key] = spec.default


def load_permalink_settings() -> None:
    params = st.query_params
    for spec in SETTING_SPECS:
        if spec.param in params and spec.key not in st.session_state:
            raw = params[spec.param]
            try:
                value = spec.cast(raw)
            except (TypeError, ValueError):
                continue
            if spec.bounds is not None:
                lo, hi = spec.bounds
                value = min(max(value, lo), hi)
            st.session_state[spec.key] = value
    if st.session_state.get("func") not in C.FUNCTIONS:
        st.session_state["func"] = C.FUNC_QUADRATIC
    if st.session_state.get("optimizer") not in C.OPTIMIZERS:
        st.session_state["optimizer"] = C.OPT_BFGS


def sync_query_params(values: dict) -> None:
    for spec in SETTING_SPECS:
        if spec.key in values:
            st.query_params[spec.param] = str(values[spec.key])


def store_from_widget(key: str) -> None:
    st.session_state[key] = st.session_state[f"widget_{key}"]


def apply_preset(preset_key: str) -> None:
    preset = C.PRESETS[preset_key]
    for field in ("func", "optimizer", "kappa", "max_iter", "seed"):
        if field in preset:
            st.session_state[field] = preset[field]
            st.session_state[f"widget_{field}"] = preset[field]


def randomize_seed() -> None:
    import random
    new_seed = random.randint(0, 999_999)
    st.session_state["seed"] = new_seed
    st.session_state["widget_seed"] = new_seed
