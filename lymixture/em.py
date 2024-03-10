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
    return utils.normalize(llhs.T, axis=0).T


def _get_params(model: models.LymphMixture) -> np.ndarray:
    """Return the params of ``model``.

    This function is very similar to the :py:meth:`.models.LymphMixture.get_params`
    method, except that it returns the mixture coefficients in the unit cube, instead
    of the simplex. Also, it just returns them as a 1D array, instead of a dictionary.
    """
    params = []
    for comp in model.components:
        params += list(comp.get_spread_params(as_dict=False))

    params += list(model.get_distribution_params(as_dict=False))
    mixture_coefs = model.get_mixture_coefs().to_numpy()
    _shape = mixture_coefs.shape
    mixture_coefs = np.apply_along_axis(utils.map_to_unit_cube, 0, mixture_coefs)
    return np.concatenate([params, mixture_coefs.flatten()])


def _set_params(model: models.LymphMixture, params: np.ndarray) -> None:
    """Set the params of ``model`` from ``params``.

    This function is very similar to the :py:meth:`.models.LymphMixture.set_params`
    method, except that it expects the mixture coefficients to from the unit cube,
    which will then be mapped to the simplex.

    Also, it does not accept a dictionary of parameters, but a 1D array.
    """
    for comp in model.components:
        params = comp.set_spread_params(*params)
    params = np.array(model.set_distribution_params(*params))

    unit_cube = params.reshape((len(model.components) - 1, len(model.subgroups)))
    simplex = np.apply_along_axis(utils.map_to_simplex, 0, unit_cube)
    model.set_mixture_coefs(simplex)


def maximization(model: models.LymphMixture, latent: np.ndarray) -> dict[str, float]:
    """Maximize ``model`` params given expectation of ``latent`` variables."""
    model.set_resps(latent)
    current_params = _get_params(model)

    def objective(params: np.ndarray) -> float:
        _set_params(model, params)
        return -model.likelihood()

    result = opt.minimize(
        fun=objective,
        x0=current_params,
        method="Powell",
        bounds=opt.Bounds(
            lb=np.zeros(shape=len(current_params)),
            ub=np.ones(shape=len(current_params)),
        ),
    )

    if result.success:
        _set_params(model, result.x)
        return model.get_params(as_dict=True)
    else:
        raise ValueError(f"Optimization failed: {result}")
