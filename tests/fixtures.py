"""Fixtures and helpers for the unit tests."""

import unittest
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scipy as sp
from lymph import diagnosis_times, modalities

from lymixture import LymphMixture
from lymixture.utils import map_to_simplex

SIMPLE_SUBSITE = ("tumor", "1", "simple_subsite")
SUBSITE = ("tumor", "1", "subsite")
MODALITIES = {
    "CT": modalities.Clinical(spec=0.81, sens=0.86),
    "FNA": modalities.Pathological(spec=0.95, sens=0.81),
}
RNG = np.random.default_rng(42)


def get_graph(size: str = "large") -> dict[tuple[str, str], list[str]]:
    """Return either a ``"small"``, a ``"medium"`` or a ``"large"`` graph."""
    if size == "small":
        return {
            ("tumor", "T"): ["II", "III", "IV"],
            ("lnl", "II"): ["III"],
            ("lnl", "III"): ["IV"],
            ("lnl", "IV"): [],
        }

    if size == "medium":
        return {
            ("tumor", "T"): ["I", "II", "III", "IV"],
            ("lnl", "I"): ["II"],
            ("lnl", "II"): ["III"],
            ("lnl", "III"): ["IV"],
            ("lnl", "IV"): [],
        }

    if size == "large":
        return {
            ("tumor", "T"): ["I", "II", "III", "IV", "V", "VII"],
            ("lnl", "I"): [],
            ("lnl", "II"): ["I", "III", "V"],
            ("lnl", "III"): ["IV", "V"],
            ("lnl", "IV"): [],
            ("lnl", "V"): [],
            ("lnl", "VII"): [],
        }

    raise ValueError(f"Unknown graph size: {size}")


def simplify_subsite(icd_code: str) -> str:
    """Only use the part of the ICD code before the decimal point."""
    return icd_code.split(".")[0]


def _create_random_frozen_dist(
    max_time: int,
    rng: np.random.Generator = RNG,
) -> np.ndarray:
    """Create a random frozen diagnosis time distribution."""
    unnormalized = rng.random(size=max_time + 1)
    return unnormalized / np.sum(unnormalized)


def _create_random_parametric_dist(
    max_time: int,
    rng: np.random.Generator = RNG,
) -> diagnosis_times.Distribution:
    """Create a binomial diagnosis time distribution with random params."""

    def _pmf(support: np.ndarray, p: float = rng.random()) -> np.ndarray:
        return sp.stats.binom.pmf(support, p=p, n=max_time + 1)

    return diagnosis_times.Distribution(
        distribution=_pmf,
        max_time=max_time,
    )


def create_random_dist(
    type_: str,
    max_time: int,
    rng: np.random.Generator = RNG,
) -> np.ndarray | Callable:
    """Create a random frozen or parametric distribution."""
    if type_ == "frozen":
        return _create_random_frozen_dist(max_time=max_time, rng=rng)

    if type_ == "parametric":
        return _create_random_parametric_dist(max_time=max_time, rng=rng)

    raise ValueError(f"Unknown distribution type: {type_}")


def get_patient_data(do_simplify_subsite: bool = True) -> pd.DataFrame:
    """Load the patient data for the tests and simplify the ICD codes."""
    patient_data = pd.read_csv(
        Path(__file__).parent / "data" / "patients.csv",
        header=[0, 1, 2],
    )

    if do_simplify_subsite:
        patient_data[SIMPLE_SUBSITE] = patient_data[SUBSITE].apply(simplify_subsite)

    return patient_data.copy()


class MixtureModelFixture(unittest.TestCase):
    """Fixture for the mixture model tests."""

    def setUp(self) -> None:
        """Create intermediate and helper objects for the tests."""
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        return super().setUp()

    def setup_rng(self, seed: int = 42):
        """Initialize random number generator with ``seed``."""
        self.rng = np.random.default_rng(seed)

    def setup_mixture_model(
        self,
        model_cls: type,
        num_components: int,
        graph_size: Literal["small", "medium", "large"] = "small",
        load_data: bool = True,
    ):
        """Initialize the fixture."""
        self.num_components = num_components
        self.model_cls = model_cls

        self.mixture_model = LymphMixture(
            model_cls=self.model_cls,
            model_kwargs={"graph_dict": get_graph(graph_size)},
            num_components=self.num_components,
        )
        if load_data:
            self.patient_data = get_patient_data()
            self.mixture_model.load_patient_data(
                self.patient_data,
                split_by=SIMPLE_SUBSITE,
            )

    def setup_responsibilities(self):
        """Initialize a set of responsibilities for the mixture model."""
        unit_cube = self.rng.uniform(
            size=(len(self.patient_data), len(self.mixture_model.components) - 1),
        )
        self.resp = np.array([map_to_simplex(line) for line in unit_cube])
