"""
Implement the EM algorithm for the mixture model.
"""
import numpy as np
from scipy import optimize as opt

from lymixture import models, utils


def expectation(model: models.LymphMixture, params: dict[str, float]) -> np.ndarray:
    """Compute expected value of latent `model`` variables given the ``params``."""
    model.set_params(**params)
    llhs = model.patient_mixture_likelihoods(log=False, marginalize=False)
    return utils.normalize(llhs, axis=1)


def _set_params(model: models.LymphMixture, params: np.ndarray) -> None:
    """Set the params of ``model`` from ``params``."""
    for comp in model.components:
        params = comp.set_params(*params)

    unit_cube = params.reshape((len(model.subgroups), len(model.components) - 1))
    simplex = np.apply_along_axis(utils.map_to_simplex, 2, unit_cube)
    model.set_mixture_coefs(simplex)


def maximization(model: models.LymphMixture, latent: np.ndarray) -> dict[str, float]:
    """Maximize ``model`` params given expectation of ``latent`` variables."""
    model.set_resps(latent)
    current_params = np.array(model.get_params(as_dict=False))

    def objective(params: np.ndarray) -> float:
        _set_params(model, params)
        return -model.likelihood()

    result = opt.minimize(
        fun=objective,
        x0=current_params,
        method="trust-constr",
        bounds=opt.Bounds(
            lb=np.zeros(shape=len(current_params)),
            ub=np.ones(shape=len(current_params)),
        ),
    )

    if result.success:
        _set_params(model, result.x)
        return model.get_params(as_dict=True)
    else:
        raise ValueError("Optimization failed.")
