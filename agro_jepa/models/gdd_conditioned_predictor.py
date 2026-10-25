"""S4A GDD-conditioned latent predictor.

Thin, explicit wrapper over :class:`S4LatentPredictor` for the conditions whose
only non-visual inputs are ``delta_days`` and/or GDD scalars.

* ``b4_gdd_R``           z_a + interval GDD          (Protocol R, no delta)
* ``b4_delta_gdd_R``     z_a + delta + interval GDD  (Protocol R, primary)
* ``b4_delta_gdd_R_tbase5``  same, T_base = 5 degC    (sensitivity)
* ``b4_delta_gdd_P``     z_a + delta + past 7/14/28-d GDD (Protocol P)

The GDD definition is pre-registered: ``GDD_d = max(T_mean_d - T_base, 0)``.
It is not fitted per genotype and there is no invented sowing date.
"""
from __future__ import annotations

from ..datasets.s4_dataset import ConditioningSpec
from .s4_conditioning import VectorNormStats
from .s4_predictor import S4LatentPredictor, S4ModelConfig
from .time_embedding import DeltaNormStats


class GDDConditionedPredictor(S4LatentPredictor):
    def __init__(
        self,
        config: S4ModelConfig,
        spec: ConditioningSpec,
        delta_norm: DeltaNormStats | None = None,
        gdd_norm: VectorNormStats | None = None,
    ) -> None:
        if spec.gdd_mode not in {"interval", "past"}:
            raise ValueError("GDDConditionedPredictor needs gdd_mode in {interval, past}")
        if spec.weather_mode != "none":
            raise ValueError("GDDConditionedPredictor must not carry a weather encoder")
        super().__init__(
            config=config,
            spec=spec,
            delta_norm=delta_norm,
            gdd_norm=gdd_norm,
            weather_norm=None,
        )

    @property
    def gdd_features(self) -> tuple[str, ...]:
        return self.spec.gdd_keys()
