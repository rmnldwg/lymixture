"""Implements the EM algorithm for the mixture model."""

from multiprocessing import Pool

import emcee
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


def maximization(model: models.LymphMixture, latent: np.ndarray) -> dict[str, float]:
    """Maximize ``model`` params given expectation of ``latent`` variables."""
    model.set_resps(latent)
    model.set_mixture_coefs(model.infer_mixture_coefs())
    current_params = _get_params(model)
    lb = np.zeros(shape=len(current_params))
    ub = np.ones(shape=len(current_params))

    def objective(params: np.ndarray) -> float:
        _set_params(model, params)
        # print(f"Optimizing with params: {params}") # DEBUG
        return -model.likelihood()

    result = opt.minimize(
        fun=objective,
        x0=current_params,
        method="Powell",
        bounds=opt.Bounds(
            lb=lb,
            ub=ub,
        ),
    )

    if result.success:
        _set_params(model, result.x)
        return model.get_params(as_dict=True)

    raise ValueError(f"Optimization failed: {result}")


def maximization_component_wise(
    model: models.LymphMixture, latent: np.ndarray
) -> dict[str, float]:
    """Maximize ``model`` params given expectation of ``latent`` variables."""
    model.set_resps(latent)
    model.set_mixture_coefs(model.infer_mixture_coefs())

    def objective(params):
        model.components[component].set_params(*params)
        return -model.component_likelihood(component=component)

    for component in range(len(model.components)):
        current_params = list(model.components[component].get_params(as_dict=False))
        lb = np.zeros(shape=len(current_params))
        ub = np.ones(shape=len(current_params))
        result = opt.minimize(
            fun=objective,
            x0=current_params,
            bounds=opt.Bounds(lb=lb, ub=ub),
            method="Powell",
        )
        if result.success:
            model.components[component].set_params(*result.x)
        else:
            raise ValueError(f"Optimization failed: {result}")

    return model.get_params(as_dict=True)


def log_prob_fn(theta, model):
    """Log probability function for the emcee sampler."""
    _set_params(model, theta)
    return model.likelihood(log=True)


def sample_model_params(
    model: models.LymphMixture,
    steps=100,
    latent=None,
) -> np.ndarray:
    """Sample ``model`` params given expectation of latent variables.

    Returns an array with the samples of the model parameters.
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
    perturbation = 1e-8 * np.random.randn(nwalkers, ndim)
    starting_points = np.ones((nwalkers, ndim)) * current_params + perturbation

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


def get_complete_samples(model: models.LymphMixture, samples: np.ndarray) -> list:
    """Return the complete set of parameters given a set of model samples."""
    parameters = []
    for i in range(samples.shape[0]):
        _set_params(model, samples[i])
        params = model.get_params(as_dict=True)
        latent = expectation(model, params)
        model.set_resps(latent)
        model.set_mixture_coefs(
            model.infer_mixture_coefs(),
        )
        parameters.append(model.get_params(as_dict=True))

    return parameters
