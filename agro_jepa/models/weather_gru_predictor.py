"""S4A weather-GRU conditioned latent predictor.

Thin, explicit wrapper over :class:`S4LatentPredictor` for the conditions that
consume a real daily weather sequence with a masked GRU.

* ``b5_delta_weather_P``  z_a + delta + GRU over the past 28 d (Protocol P)
* ``b5_delta_weather_R``  z_a + delta + GRU over ``(t_a, t_b]`` (Protocol R)

Protocol P is the deployable one: the sequence ends strictly before ``t_a``.
Protocol R is an oracle interval analysis and must never be reported as a
forecast.  Channels are the five confirmed-unit S1 variables; ``P_sum_raw`` is
excluded because its unit is unresolved.
"""
from __future__ import annotations

from ..datasets.s4_dataset import ConditioningSpec
from .s4_conditioning import VectorNormStats
from .s4_predictor import S4LatentPredictor, S4ModelConfig
from .time_embedding import DeltaNormStats


class WeatherGRUPredictor(S4LatentPredictor):
    def __init__(
        self,
        config: S4ModelConfig,
        spec: ConditioningSpec,
        delta_norm: DeltaNormStats | None = None,
        weather_norm: VectorNormStats | None = None,
    ) -> None:
        if spec.weather_mode not in {"past", "interval"}:
            raise ValueError("WeatherGRUPredictor needs weather_mode in {past, interval}")
        if spec.gdd_mode != "none":
            raise ValueError("WeatherGRUPredictor must not carry a GDD encoder")
        super().__init__(
            config=config,
            spec=spec,
            delta_norm=delta_norm,
            gdd_norm=None,
            weather_norm=weather_norm,
        )

    @property
    def weather_window(self) -> str:
        return self.spec.weather_mode
