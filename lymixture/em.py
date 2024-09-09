"""Implements the EM algorithm for the mixture model."""

import logging
from collections.abc import Callable
from multiprocessing import Pool

import emcee
import numpy as np
from scipy import optimize as opt

from lymixture import models, utils

logger = logging.getLogger(__name__)


def expectation(model: models.LymphMixture, params: dict[str, float]) -> np.ndarray:
    """Compute expected value of latent ``model`` variables given the ``params``."""
    model.set_params(**params)
    llhs = model.patient_mixture_likelihoods(log=False, marginalize=False)
    return utils.normalize(llhs.T, axis=0).T


def _get_params(model: models.LymphMixture) -> np.ndarray:
    """Return the params of ``model``.

    This function is very similar to the :py:meth:`.models.LymphMixture.get_params`
    method. Also, it just returns them as a 1D array, instead of a dictionary.
    """
    params = []
    if model.universal_p:
        for comp in model.components:
            params += list(comp.get_spread_params(as_dict=False))

        params += list(model.get_distribution_params(as_dict=False))
    else:
        for comp in model.components:
            params += list(comp.get_params(as_dict=False))
    return params


def _set_params(model: models.LymphMixture, params: np.ndarray) -> None:
    """Set the params of ``model`` from ``params``.

    This function is very similar to the :py:meth:`.models.LymphMixture.set_params`
    method.

    Also, it does not accept a dictionary of parameters, but a 1D array.
    """
    if model.universal_p:
        for comp in model.components:
            params = comp.set_spread_params(*params)
        params = np.array(model.set_distribution_params(*params))
    else:
        for comp in model.components:
            params = comp.set_params(*params)
        params = np.array(params)


def init_callback() -> Callable:
    """Return a function that logs the optimization progress."""
    iteration = 0

    def log_optimization(xk):
        nonlocal iteration
        logger.debug(f"Iteration {iteration} with params: {xk}")
        iteration += 1

    return log_optimization


def neg_complete_component_llh(
    params: np.ndarray,
    model: models.LymphMixture,
    component: int,
) -> float:
    """Return the negative complete log likelihood of ``component`` in ``model``."""
    model.components[component].set_params(*params)
    result = -model.complete_data_likelihood(component=component)
    logger.debug(f"Component {component} with params {params} has llh {result}")
    return result


def maximization(
    model: models.LymphMixture,
    latent: np.ndarray,
) -> dict[str, float]:
    """Maximize ``model`` params given expectation of ``latent`` variables."""
    maximized_mixture_coefs = model.infer_mixture_coefs(new_resps=latent)
    model.set_mixture_coefs(maximized_mixture_coefs)
    model.normalize_mixture_coefs()

    for i, component in enumerate(model.components):
        current_params = list(component.get_params(as_dict=False))
        lb = np.zeros(shape=len(current_params))
        ub = np.ones(shape=len(current_params))

        result = opt.minimize(
            fun=neg_complete_component_llh,
            args=(model, i),
            x0=current_params,
            bounds=opt.Bounds(lb=lb, ub=ub),
            method="Powell",
            callback=init_callback(),
        )

        if result.success:
            component.set_params(*result.x)
        else:
            raise RuntimeError(f"Optimization failed: {result}")

    return model.get_params(as_dict=True)


def log_prob_fn(theta, model):
    """Log probability function for the emcee sampler."""
    try:
        _set_params(model, theta)
    except ValueError:
        return -np.inf

    return model.likelihood(log=True)


def sample_model_params(
    model: models.LymphMixture,
    steps: int = 100,
    latent: np.ndarray | None = None,
    spread_size: float = 1e-5,
) -> np.ndarray:
    """Sample params of the components of the ``model`` given latent variables.

    The samples are drawn from the complete data log-likelihood of the model.
    """
    # NOTE: Right now the samples are very close to each other, such that the resulting
    #       differences in the mixture parameters are very small -> There is probably
    #       an error here.
    if latent is None:
        latent = model.get_resps()

    model.set_resps(latent)
    model.set_mixture_coefs(model.infer_mixture_coefs())
    current_params = _get_params(model)

    ndim = len(current_params)
    nwalkers = 5 * ndim
    perturbation = spread_size * np.random.randn(nwalkers, ndim)
    starting_points = current_params + perturbation

    starting_points = np.where(
        starting_points < 0,
        0,
        starting_points,
    )
    starting_points = np.where(
        starting_points > 1,
        1,
        starting_points,
    )

    # Pass model as an additional argument to log_prob_fn
    with Pool() as pool:
        original_sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_prob_fn,
            args=(model,),  # Pass model here
            pool=pool,
        )
        original_sampler.run_mcmc(
            initial_state=starting_points,
            nsteps=steps,
            progress=True,
        )

    return original_sampler.get_chain(discard=0, thin=1, flat=True)


def get_complete_samples(
    model: models.LymphMixture,
    samples: np.ndarray,
) -> list[dict[str, float]]:
    """For each parameter sample, compute corresponding mixture coefficients."""
    params = []

    for sample in samples:
        _set_params(model, sample)
        latent = model.patient_mixture_likelihoods(log=False, marginalize=False)
        latent = utils.normalize(latent.T, axis=0).T

        model.set_resps(latent)
        mixture_coefs = model.infer_mixture_coefs()
        model.set_mixture_coefs(mixture_coefs)
        model.normalize_mixture_coefs()

        params.append(model.get_params())

    return params
