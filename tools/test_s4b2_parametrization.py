#!/usr/bin/env python3
"""S4B.2 parametrization audit + condition-semantics test.

Verifies, on CPU, that:
* b2_no_time builds with predictor_input_dim == 928 and 0 condition params;
* b2_no_time_constant_delta and b3_delta build with predictor_input_dim == 1088,
  n_condition_parameters == 72,730, n_trainable_parameters == 53,882,396;
* b2_no_time has no delta path; B2-C carries a fixed_delta_days buffer of 9.0;
  b3 receives the real per-sample delta_days;
* b2 does not accept delta; B2-C/b3 share the same init state up to the (buffer)
  fixed value so the only difference is the constant-delta mechanism.

Run: pytest tools/test_s4b2_parametrization.py -q
(or python3 tools/test_s4b2_parametrization.py for a standalone run).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agro_jepa.models.s4_predictor import (  # noqa: E402
    S4ModelConfig,
    build_s4_predictor,
    conditioning_spec,
    has_fixed_delta,
)


def _model(condition: dict):
    from agro_jepa.models.fip_dino_wm import move_attention_masks
    m = _build(condition)
    move_attention_masks(m.predictor, torch.device("cpu"))
    return m


def _build(condition: dict):
    cfg = S4ModelConfig.from_dict({
        "num_patches": 256, "visual_dim": 768, "depth": 6, "heads": 16,
        "mlp_dim": 2048, "dropout": 0.1, "emb_dropout": 0.0, "pool": "mean",
        "action_emb_dim": 10, "num_action_repeat": 16, "proprio_emb_dim": 10,
        "num_proprio_repeat": 16,
        "delta": {"n_frequencies": 8, "base_frequency": 2.0, "hidden_dim": 256,
                  "output_dim": 10, "dropout": 0.0, "use_learned_scale": True},
    })
    spec = conditioning_spec(condition)
    delta_norm = None
    if spec.use_delta:
        from agro_jepa.models.time_embedding import DeltaNormStats
        delta_norm = DeltaNormStats(mean=10.25, std=3.68, min=5.0, max=16.0, n_samples=10)
        if spec.delta_mode == "constant":
            from dataclasses import replace
            spec = replace(spec, fixed_delta_days=float(condition["fixed_delta_days"]))
    return build_s4_predictor(cfg, spec, delta_norm=delta_norm)


def test_b2_no_time_shapes():
    m = _model({"mode": "trained", "use_delta": False, "gdd_mode": "none",
                "weather_mode": "none"})
    assert m.has_condition is False
    assert m.config.predictor_dim(False) == 928
    assert m.n_trainable_parameters() == 45_898_196
    assert m.n_condition_parameters() == 0
    assert "delta" not in m.parts


def test_b2c_and_b3_matched_dims():
    b2c = _model({"mode": "trained", "use_delta": True, "delta_mode": "constant",
                  "fixed_delta_days": 9.0, "gdd_mode": "none", "weather_mode": "none"})
    b3 = _model({"mode": "trained", "use_delta": True, "delta_mode": "sample",
                 "gdd_mode": "none", "weather_mode": "none"})
    assert b2c.config.predictor_dim(True) == 1088
    assert b3.config.predictor_dim(True) == 1088
    assert b2c.n_trainable_parameters() == 53_882_396
    assert b3.n_trainable_parameters() == 53_882_396
    assert b2c.n_condition_parameters() == 72_730
    assert b3.n_condition_parameters() == 72_730
    assert has_fixed_delta(b2c) and not has_fixed_delta(b3)


def test_b2c_fixed_delta_value():
    b2c = _model({"mode": "trained", "use_delta": True, "delta_mode": "constant",
                  "fixed_delta_days": 9.0, "gdd_mode": "none", "weather_mode": "none"})
    assert float(b2c.fixed_delta_days) == 9.0


def test_b2_ignores_delta():
    """b2_no_time has no delta path: its output cannot depend on delta_days."""
    m = _model({"mode": "trained", "use_delta": False, "gdd_mode": "none",
                "weather_mode": "none"})
    m.eval()
    assert "delta" not in m.parts
    torch.manual_seed(0)
    z = torch.randn(2, 256, 768)
    out_a = m(z)
    out_b = m(z, delta_days=torch.tensor([5.0, 12.0]))
    assert out_a.shape == (2, 256, 768)
    assert torch.allclose(out_a, out_b), "b2_no_time output depended on delta_days"


def test_b3_accepts_sample_delta():
    m = _model({"mode": "trained", "use_delta": True, "delta_mode": "sample",
                "gdd_mode": "none", "weather_mode": "none"})
    z = torch.randn(2, 256, 768)
    z_out = m(z, delta_days=torch.tensor([5.0, 12.0]))
    assert z_out.shape == (2, 256, 768)


def test_b2c_b3_init_match_up_to_fixed_buffer():
    """Same seed -> identical trainable init; only the fixed-delta buffer differs."""
    def state_of(condition):
        torch.manual_seed(1234)   # same seed immediately before each build
        m = _model(condition)
        return {k: v.clone() for k, v in m.state_dict().items()
                if "fixed_delta_days" not in k}
    s_b2c = state_of({"mode": "trained", "use_delta": True, "delta_mode": "constant",
                      "fixed_delta_days": 9.0, "gdd_mode": "none", "weather_mode": "none"})
    s_b3 = state_of({"mode": "trained", "use_delta": True, "delta_mode": "sample",
                     "gdd_mode": "none", "weather_mode": "none"})
    assert set(s_b2c) == set(s_b3)
    mism = [k for k in s_b2c if not torch.equal(s_b2c[k], s_b3[k])]
    assert mism == [], f"non-buffer init mismatch: {mism}"


if __name__ == "__main__":
    test_b2_no_time_shapes()
    test_b2c_and_b3_matched_dims()
    test_b2c_fixed_delta_value()
    test_b2_ignores_delta()
    test_b3_accepts_sample_delta()
    test_b2c_b3_init_match_up_to_fixed_buffer()
    print("all S4B.2 parametrization checks passed")
