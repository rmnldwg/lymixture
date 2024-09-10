"""
Generate synthetic data for testing.
"""

import argparse
from collections import namedtuple
from pathlib import Path
from typing import Any, Literal
from collections.abc import Callable

import numpy as np
import pandas as pd
from lymph import models

from lymixture import utils

Modality = namedtuple("Modality", ["spec", "sens"])


GRAPH_DICT = {
    ("tumor", "T"): ["II", "III"],
    ("lnl", "II"): ["III"],
    ("lnl", "III"): [],
}
MODALITIES = {
    "path": Modality(spec=0.9, sens=0.9),
}
DISTRIBUTIONS = {
    "early": utils.binom_pmf(k=np.arange(11), n=10, p=0.3),
    "late": utils.late_binomial,
}
PARAMS_C1 = {
    "TtoII_spread": 0.05,
    "TtoIII_spread": 0.25,
    "IItoIII_spread": 0.5,
    "late_p": 0.5,
}
PARAMS_C2 = {
    "TtoII_spread": 0.25,
    "TtoIII_spread": 0.05,
    "IItoIII_spread": 0.1,
    "late_p": 0.5,
}
SUBSITE_COL = ("tumor", "1", "subsite")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-1",
        "--num-c1",
        type=int,
        default=100,
        help="Number of samples for the first dataset.",
    )
    parser.add_argument(
        "-2",
        "--num-c2",
        type=int,
        default=100,
        help="Number of samples for the second dataset.",
    )
    parser.add_argument(
        "-3",
        "--num-c3",
        type=int,
        default=100,
        help="Number of samples for the third dataset.",
    )
    parser.add_argument(
        "-m",
        "--mix",
        type=float,
        default=0.5,
        help="Mixing ratio for the third dataset.",
    )
    parser.add_argument(
        "-t",
        "--tstage-ratio",
        type=float,
        default=0.6,
        help="Ratio of early vs late stage patients.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="mixture.csv",
        help="Output file for the mixture dataset.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Seed for the random number generator.",
    )
    return parser


ModalityDict = dict[str, dict[str, float | Literal["clinical", "pathological"]]]


def create_model(
    model_kwargs: dict[str, Any] | None = None,
    modalities: ModalityDict | None = None,
    distributions: dict[str, list[float] | Callable] | None = None,
) -> models.Unilateral:
    """Create a model to draw patients from."""
    model = models.Unilateral(**(model_kwargs or {"graph_dict": GRAPH_DICT}))

    for name, modality in (modalities or MODALITIES).items():
        model.set_modality(name, modality.spec, modality.sens)

    for t_stage, dist in (distributions or DISTRIBUTIONS).items():
        model.set_distribution(t_stage, dist)

    return model


def draw_datasets(
    model: models.Unilateral,
    num_c1: int,
    num_c2: int,
    num_c3: int,
    tstage_ratio: float,
    mix: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw patients for the three datasets."""
    model.set_params(**PARAMS_C1)
    c1_data = model.draw_patients(
        num=num_c1 + int(num_c3 * mix),
        stage_dist=[tstage_ratio, 1 - tstage_ratio],
        rng=rng,
    )
    model.set_params(**PARAMS_C2)
    c2_data = model.draw_patients(
        num=num_c2 + int(num_c3 * (1 - mix)),
        stage_dist=[tstage_ratio, 1 - tstage_ratio],
        rng=rng,
    )
    c3_data = pd.concat(
        [
            c1_data.iloc[num_c1:],
            c2_data.iloc[num_c2:],
        ],
        ignore_index=True,
        axis=0,
    )
    c1_data = c1_data.iloc[:num_c1]
    c2_data = c2_data.iloc[:num_c2]

    c1_data[SUBSITE_COL] = "c1"
    c2_data[SUBSITE_COL] = "c2"
    c3_data[SUBSITE_COL] = "c3"

    return pd.concat([c1_data, c2_data, c3_data], ignore_index=True, axis=0)


def main() -> None:
    """Generate three datasets: Two with distinct params and one as a mix of both."""
    parser = create_parser()
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    model = create_model()
    mixture_data = draw_datasets(
        model=model,
        num_c1=args.num_c1,
        num_c2=args.num_c2,
        num_c3=args.num_c3,
        tstage_ratio=args.tstage_ratio,
        mix=args.mix,
        rng=rng,
    )
    mixture_data.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
