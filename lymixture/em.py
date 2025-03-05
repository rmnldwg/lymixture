"""Implements the `EM algorithm`_ for the mixture model.

Using the class :py:class:`.models.LymphMixture` and its methods, this module provides
functions to compute the expectation and maximization steps of the `EM algorithm`_.

.. _EM algorithm: https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm"""

import logging
from collections.abc import Callable
from multiprocessing import Pool

import emcee
import numpy as np
from scipy import optimize as opt
import pandas as pd

from lymixture import models, utils

logger = logging.getLogger(__name__)


def _get_params(model: models.LymphMixture) -> np.ndarray:
    """Return the params of ``model``.

    .. seealso::
        This function is very similar to the :py:meth:`.models.LymphMixture.get_params`
        method. Except that it does not accept a dictionary of parameters, but only a 1D
        array.
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

    .. seealso::
        This function is very similar to the :py:meth:`.models.LymphMixture.set_params`
        method. Except that it does not accept a dictionary of parameters, but only a 1D
        array.
    """
    if model.universal_p:
        for comp in model.components:
            params = comp.set_spread_params(*params)
        params = np.array(model.set_distribution_params(*params))
    else:
        for comp in model.components:
            params = comp.set_params(*params)
        params = np.array(params)


def expectation(model: models.LymphMixture, params: dict[str, float]) -> np.ndarray:
    """Compute expected value of latent ``model`` variables given the ``params``.

    This marks the E-step of the famous `EM algorithm`_. The returned expected values
    are also often called responsibilities.

    .. _EM algorithm: https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm
    """
    model.set_params(**params)
    llhs = model.patient_mixture_likelihoods(log=False, marginalize=False)
    return utils.normalize(llhs.T, axis=0).T


def init_callback() -> Callable:
    """Return a function that logs the optimization progress."""
    iteration = 0

    def log_optimization(xk):
        nonlocal iteration
        logger.debug(f"Iteration {iteration} with params: {xk}")
        iteration += 1

    return log_optimization


def _neg_complete_component_llh(
    params: np.ndarray,
    model: models.LymphMixture,
    component: int,
) -> float:
    """Return the negative complete log likelihood of ``component`` in ``model``.

    This function is used in the M-step of the EM algorithm.
    """
    try:
        model.components[component].set_params(*params) #we can set it for all of them here in theory as well. But I am short in time so I will fix this later
    except ValueError:
        return np.inf
    result = -model.complete_data_likelihood(component=component)
    logger.debug(f"Component {component} with params {params} has llh {result}")
    return result


def maximization(
    model: models.LymphMixture,
    latent: np.ndarray,
) -> dict[str, float]:
    """Maximize ``model`` params given expectation of ``latent`` variables.

    This is the corresponding M-step to the :py:func:`.expectation` of the
    `EM algorithm`_. It first maximizes the mixture coefficients analytically and then
    optimizes the model parameters of all components sequentially.

    .. _EM algorithm: https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm
    """
    maximized_mixture_coefs = model.infer_mixture_coefs(new_resps=latent)
    model.set_mixture_coefs(maximized_mixture_coefs)
    model.normalize_mixture_coefs()

    for i, component in enumerate(model.components):
        current_params = list(component.get_params(as_dict=False))
        lb = np.zeros(shape=len(current_params))
        ub = np.ones(shape=len(current_params))

        result = opt.minimize(
            fun=_neg_complete_component_llh,
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


def log_prob_fn_fixed_mixture(theta, model):
    """
    Computes the log-probability of the model given its parameters, excluding mixture coefficients.

    This function calculates the log-probability of a mixture model based on the provided parameters 
    (`theta`), assuming that mixture coefficients remain fixed. It ensures that the parameter values 
    are within the valid range [0, 1], and returns negative infinity (`-inf`) if any parameter is out 
    of bounds.

    Args:
        theta (numpy.ndarray): Array of model parameters, excluding mixture coefficients.
        model (lymixture.models): The mixture model instance for which the log-probability is computed.

    Returns:
        float: The log-probability of the model if parameters are valid, or `-inf` if parameters 
        are out of bounds.

    Notes:
        - This function does not modify or include mixture coefficients in `theta`; these are 
          assumed to remain unchanged.
        - The `_set_params` function is used to update the model parameters before computing 
          the likelihood.
    """
    lower_bounds = np.zeros(len(theta))
    upper_bounds = np.ones(len(theta)) 
    # Check if the parameters are within bounds
    if np.any(theta < lower_bounds) or np.any(theta > upper_bounds):
        return -np.inf  # Return -infinity if out of bounds
    _set_params(model,theta)
    return model.likelihood(log=False)

def log_prob_fn(theta, model):
    """
    Computes the log-probability of the model given its parameters.

    This function evaluates the log-probability of the provided mixture model based on the 
    given parameter values (`theta`). It ensures that parameters stay within predefined bounds 
    (0 to 1). If any parameter is out of bounds, the function returns negative infinity (`-inf`).

    Args:
        theta (numpy.ndarray): Array of model parameters, including mixture coefficients.
        model (lymixture.models): The mixture model instance for which the log-probability is computed.

    Returns:
        float: The log-probability of the model if parameters are valid, or `-inf` if parameters 
        are out of bounds.

    Notes:
        - The `theta` array includes mixture parameters, which are not sampled from a simplex. 
          This behavior could be extended to enforce simplex constraints if required.
    """
    lower_bounds = np.zeros(len(theta))
    upper_bounds = np.ones(len(theta)) 

    # Check if the parameters are within bounds
    if np.any(theta < lower_bounds) or np.any(theta > upper_bounds):
        return -np.inf  # Return -infinity if out of bounds
    model.set_params(*theta)
    return model.likelihood(log=True, use_complete = True)


def sample_fixed_mixture(model, steps=100, latent=None, filename = 'chain_fixed_mix.hdf5', continue_sampling = False) -> np.ndarray:
    """
    Samples the parameters of a mixture model, excluding mixture coefficients.

    This function performs MCMC sampling for the parameters of a mixture model while 
    keeping the mixture coefficients fixed. It allows the specification of latent parameters 
    and offers options to either start a new sampling session or continue from an existing 
    HDF5 backend file.

    Args:
        model (lymixture.models): The mixture model with data and parameters initialized.
        steps (int, optional): Number of MCMC sampling steps to perform. Defaults to 100.
        latent (pandas.DataFrame, optional): Assignment of latent variables. If None, the 
            latent variables are determined using `model.get_resps()`. Defaults to None.
        filename (str, optional): Path to the HDF5 file for storing the sampler's backend. 
            Defaults to 'chain_fixed_mix.hdf5'.
        continue_sampling (bool, optional): If True, sampling continues from the current state 
            in the backend file. If False, the backend is reset, and sampling starts anew. 
            Defaults to False.

    Returns:
        emcee.backends.HDFBackend: The backend storing the MCMC chain and metadata.
        numpy.ndarray: A flat array of sampled parameter chains.

    Notes:
        - The model's responsibilities (`resps`) and mixture coefficients are updated based 
          on the provided or computed latent parameters.
        - Mixture coefficients are fixed during the sampling process.
        - The function initializes an `emcee.EnsembleSampler` with a fixed mixture coefficient 
          log-probability function (`log_prob_fn_fixed_mixture`) and uses multiprocessing to 
          parallelize sampling.

    Example:
        >>> backend, chain = sample_fixed_mixture(my_model, steps=500, filename='my_fixed_mix.hdf5')

    """
    if latent is None:
        latent = model.get_resps()
    model.set_resps(latent)
    maximized_mixture_coefs = model.infer_mixture_coefs(new_resps=latent)
    model.set_mixture_coefs(maximized_mixture_coefs)
    current_params = _get_params(model)
    
    ndim = len(current_params)
    nwalkers = 5 * ndim
    perturbation = 1e-6 * np.random.randn(nwalkers, ndim)
    backend = emcee.backends.HDFBackend(filename)
    if continue_sampling is False:
        starting_points = np.ones((nwalkers, ndim)) * current_params + perturbation
        backend.reset(nwalkers, ndim)
    else:
        starting_points = None
    # Pass model as an additional argument to log_prob_fn
    with Pool() as pool:
        print(f"Number of cores (workers) used by the emcee sampler: {pool._processes}")

        original_sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob_fn_fixed_mixture,
            args=(model,),  # Pass model here
            pool=pool,backend=backend
        )
        original_sampler.run_mcmc(initial_state=starting_points, nsteps=steps, progress=True)

    return backend, original_sampler.get_chain(discard=0, thin=1, flat=True)


def sample_model_params(model, steps = 100, latent = None, filename = 'chain_fixed_latent.hdf5', continue_sampling = False):
    """
    Samples the parameters of a mixture model given expectations of latent variables.

    This function performs Markov Chain Monte Carlo (MCMC) sampling of the parameters 
    of a provided mixture model. It allows setting latent parameters and provides options 
    to either start sampling from scratch or continue from a previous state stored in an HDF5 file.

    Args:
        model (lymixture.models): The mixture model with data and parameters already set.
        steps (int, optional): The number of MCMC sampling steps. Defaults to 100.
        latent (pandas.DataFrame, optional): Assignment of latent variables. If None, the latent 
            variables are computed using `model.get_resps()`. Defaults to None.
        filename (str, optional): Path to the HDF5 file used for storing the sampler backend. 
            Defaults to 'chain_fixed_latent.hdf5'.
        continue_sampling (bool, optional): If True, sampling continues without resetting the backend. 
            If False, the backend is reset, and sampling starts anew. Defaults to False.

    Returns:
        emcee.backends.HDFBackend: The backend storing the MCMC chain and metadata.
        numpy.ndarray: A flat array of sampled parameter chains.

    Notes:
        - The model's responsibilities (`resps`) and mixture coefficients are updated based 
          on the provided or computed latent parameters.
        - The function initializes an `emcee.EnsembleSampler` for MCMC sampling and uses a 
          multiprocessing pool to parallelize the computations.

    Example:
        >>> backend, chain = sample_model_params(my_model, steps=500, filename='my_chain.hdf5')
    """
    if latent is None:
        latent = model.get_resps()
    model.set_resps(latent)
    model.set_mixture_coefs(model.infer_mixture_coefs())
    current_params = list(model.get_params(as_dict = False))
    
    ndim = len(current_params)
    nwalkers = 5 * ndim
    perturbation = 1e-6 * abs(np.random.randn(nwalkers, ndim))
    backend = emcee.backends.HDFBackend(filename)

    if continue_sampling is False:
        starting_points = np.ones((nwalkers, ndim)) * current_params + perturbation
        starting_points[starting_points > 1] = 1 - perturbation[starting_points > 1]
        backend.reset(nwalkers, ndim)
    else:
        starting_points = None
    
    with Pool() as pool:
        original_sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob_fn,
            args=(model,),
            pool=pool, backend = backend
        )
        original_sampler.run_mcmc(initial_state=starting_points, nsteps=steps, progress=True)

    return backend, original_sampler.get_chain(discard=0, thin=1, flat=True)


def complete_latent_likelihood(theta, model):
    """
    Computes the complete data log-likelihood for a mixture model with given latent variable assignments.

    This function evaluates the log-likelihood of the mixture model using a provided set of latent 
    variable assignments (`theta`). The assignments are set as the responsibilities (`resps`) of the 
    model before computing the likelihood.

    Args:
        theta (numpy.ndarray): A 2D array representing the latent variable responsibilities, 
            where rows correspond to data points and columns to components.
        model (lymixture.models): The mixture model instance for which the log-likelihood is computed.

    Returns:
        float: The complete data log-likelihood of the model given the latent variable assignments.
    """
    df = model.get_resps()
    sampled_df = pd.DataFrame(theta, index=df.index, columns=df.columns)  # Create a DataFrame of zeros
    model.set_resps(sampled_df)
    return model.likelihood(log=True, use_complete = True)


def mh_latent_sampler_per_patient_2_component(model, temperature=None):
    """
    Performs Metropolis-Hastings sampling for latent variables on a per-patient basis for 2 components.

    This function implements a basic Metropolis-Hastings (MH) sampler to update the latent 
    variables (responsibilities) of a mixture model for individual patients. It swaps the 
    latent variable assignments for two components, evaluates the log-acceptance ratio, and 
    accepts or rejects the proposed changes based on the Metropolis criterion.

    Args:
        model (lymixture.models): The mixture model instance, initialized with data and parameters.
        temperature (float, optional): A scaling factor applied to the acceptance ratio to 
            adjust the sampling behavior. Lower temperatures make the sampler stricter. Defaults to 0.5.

    Returns:
        tuple:
            - pandas.DataFrame: The latent variable responsibilities (`current_position`) 
              before the sampling step.
            - float: The log-probability of the model before the sampling step.

    Notes:
        - The sampler works by proposing a swap of responsibilities between two components 
          for each patient and calculating the acceptance ratio using the patient-specific 
          mixture likelihoods.
        - Accepted swaps are updated in the latent variable matrix (`accepted_position`).
        - The current and new log-probabilities are computed using the provided `log_prob_fn`.
        - This function is desgined for a full AIP algorithm but is not used due to long computation times.

    Example:
        >>> current_resps, current_log_prob = mh_sampler_per_patient(my_model, log_prob_fn)
        >>> print("Log probability before sampling:", current_log_prob)

    """
    if temperature is None:
        temperature = 0.5
    current_position = model.get_resps()
    new_position = current_position.copy()
    accepted_position = current_position.copy()
    current_log_prob = complete_latent_likelihood(current_position, model)
    new_position.iloc[:, [0, 1]] = current_position.iloc[:, [1, 0]].to_numpy()
    
    current_assignments = np.argmax(np.array(current_position), axis=1)
    new_assignments = np.argmax(np.array(new_position), axis=1)

    log_acceptance_ratio = (
        (model.patient_mixture_likelihoods(log=True)[np.arange(len(new_assignments)), new_assignments] -
         model.patient_mixture_likelihoods(log=True)[np.arange(len(current_assignments)), current_assignments]) / temperature
    )
    accept_ratio = np.exp(log_acceptance_ratio)
    accept_thresholds = np.random.rand(len(accept_ratio))
    accepted_indices = np.where(accept_thresholds < accept_ratio)[0]
    accepted_position.iloc[accepted_indices, [0, 1]] = current_position.iloc[accepted_indices, [1, 0]].to_numpy()
    model.set_resps(accepted_position)
    new_log_prob = complete_latent_likelihood(accepted_position, model)
    print(len(accepted_indices), 'swaps accepted')
    return current_position, current_log_prob


def aip_sampling_algorithm(
    mixture, 
    IP_rounds=4000, 
    n_steps_params=1, 
    temperature_schedule=None, 
    params_filename="../../params_samples.hdf5"
    ):
    """
    Performs Alternating Iterative Posterior (AIP) sampling for a mixture model.

    This function alternates between sampling latent variables and model parameters 
    to approximate the posterior distribution of a mixture model. The AIP algorithm 
    integrates Metropolis-Hastings (MH) sampling for latent variables and a parameter 
    sampler initialized with `emcee`. This is computationally intensive and may take
    a long time to converge and is therefore only used for toy problems.

    Args:
        mixture (lymixture.models): The mixture model instance, initialized with data and parameters.
        IP_rounds (int, optional): Number of iterations for the AIP sampling loop. Defaults to 4000.
        n_steps_params (int, optional): Number of steps for parameter sampling in each iteration. Defaults to 1.
        params_filename (str, optional): File path for storing the parameter sampler's backend data. 
            Defaults to "../../params_samples.hdf5".

    Returns:
        dict: A dictionary containing:
            - "params_samples" (list): Samples of model parameters.
            - "latent_samples" (list): Samples of latent variables.
            - "complete_likelihoods" (list): Complete data log-likelihoods across iterations.
            - "incomplete_likelihoods" (list): Incomplete data log-likelihoods across iterations.
            - "number_of_swaps" (list): Number of swaps in latent variables between iterations.
    """
    # Initialization
    n_dim_params = len(mixture.get_params())
    n_walkers_params = 5 * n_dim_params

    # Lists to store results
    params_samples = []
    latent_samples = []
    complete_likelihoods = []
    incomplete_likelihoods = []
    number_of_swaps = []

    # Initialize latent variables
    starting_latent = mixture.get_resps()
    starting_latent.iloc[:, 0] = np.random.choice([0, 1], len(starting_latent))
    starting_latent.iloc[:, 1] = 1 - starting_latent.iloc[:, 0]
    mixture.set_resps(starting_latent)

    # Initialize parameter sampler
    backend_params, params_samples = sample_model_params(mixture, steps = 1, filename = params_filename, continue_sampling = False)

    # Initial samples
    latent_samples.append(mixture.get_resps())
    params_samples.append(mixture.get_params(as_dict=False))

    for round in range(IP_rounds):
        # Determine temperature
        if temperature_schedule is None:
            temperature = 1 - round / IP_rounds + 0.05
        else:
            temperature = temperature_schedule(round)

        # Latent sampling
        new_latent, current_prob = mh_latent_sampler_per_patient_2_component(
            mixture, complete_latent_likelihood, temperature
        )
        latent_samples.append(new_latent)

        # Parameter sampling
        backend_params, params_samples = sample_model_params(mixture, steps = n_steps_params, filename = params_filename, continue_sampling = True)
        new_params_samples = backend_params.get_chain(discard=0, thin=1, flat=False)

        # Extract the last parameter sample and update
        samples_flat = new_params_samples.reshape(-1, n_dim_params)  # Flatten correctly
        params_samples.append(samples_flat[-1])
        mixture.set_params(*samples_flat[-1])

        # Compute likelihoods for diagnostics
        complete_likelihoods.append(mixture.likelihood(use_complete=True))
        incomplete_likelihoods.append(mixture.likelihood(use_complete=False))

        if round != 0:
            number_of_swaps.append(
                abs(latent_samples[-1] - latent_samples[-2]).sum().sum() / 2
            )

        print(
            f"Complete likelihood: {complete_likelihoods[-1]}, "
            f"Incomplete likelihood: {incomplete_likelihoods[-1]}"
        )

    return {
        "params_samples": params_samples,
        "latent_samples": latent_samples,
        "complete_likelihoods": complete_likelihoods,
        "incomplete_likelihoods": incomplete_likelihoods,
        "number_of_swaps": number_of_swaps,
    }
