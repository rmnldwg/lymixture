"""
This module defines the class wrapping the base model and composing the mixture model
likelihood from the components and subgroups in the data.
"""
# pylint: disable=logging-fstring-interpolation

import logging
import warnings
from typing import Any, Iterable

import lymph
import numpy as np
import pandas as pd
from lymph import diagnose_times, modalities, types
from lymph.helper import flatten, popfirst, unflatten_and_split

from lymixture.utils import RESP_COL, T_STAGE_COL, join_with_responsibilities

pd.options.mode.copy_on_write = True
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)



class LymphMixture(
    diagnose_times.Composite,   # NOTE: The order of inheritance must be the same as the
    modalities.Composite,       #       order in which the respective __init__ methods
    types.Model,                #       are called.
):
    """Class that handles the individual components of the mixture model.
    """
    def __init__(
        self,
        model_cls: type = lymph.models.Unilateral,
        model_kwargs: dict[str, Any] | None = None,
        num_components: int = 2,
    ):
        """Initialize the mixture model.

        The mixture will be based on the given ``model_cls`` (which is instantiated with
        the ``model_kwargs``), and will have ``num_components``.
        """
        if model_kwargs is None:
            model_kwargs = {"graph_dict": {
                ("tumor", "T"): ["II", "III"],
                ("lnl", "II"): ["III"],
                ("lnl", "III"): [],
            }}

        if not issubclass(model_cls, lymph.models.Unilateral):
            raise NotImplementedError(
                "Mixture model only implemented for `Unilateral` model."
            )

        self._model_cls = model_cls
        self._model_kwargs = model_kwargs
        self._mixture_coefs = None

        self.subgroups: dict[str, model_cls] = {}
        self.components: list[model_cls] = self._create_components(num_components)

        diagnose_times.Composite.__init__(
            self,
            distribution_children=dict(enumerate(self.components)),
            is_distribution_leaf=False,
        )
        logger.info(
            f"Created LymphMixtureModel based on {model_cls} model with "
            f"{num_components} components."
        )


    def _create_components(self, num_components: int) -> list[Any]:
        """Initialize the component parameters and assignments."""
        components = []
        for _ in range(num_components):
            components.append(self._model_cls(**self._model_kwargs))

        return components


    def _create_empty_mixture_coefs(self) -> pd.DataFrame:
        nan_array = np.empty((len(self.components), len(self.subgroups)))
        nan_array[:] = np.nan
        return pd.DataFrame(
            nan_array,
            index=range(len(self.components)),
            columns=self.subgroups.keys(),
        )


    @property
    def is_trinary(self) -> bool:
        """Check if the model is trinary."""
        if not (
            all(sub.is_trinary for sub in self.subgroups.values())
            == all(comp.is_trinary for comp in self.components)
        ):
            raise ValueError("Subgroups & components not all trinary or not all binary.")

        return self.components[0].is_trinary


    def get_mixture_coefs(
        self,
        component: int | None = None,
        subgroup: str | None = None,
        normalize: bool = True,
    ) -> float | pd.Series | pd.DataFrame:
        """Get mixture coefficients for the given ``subgroup`` and ``component``.

        The mixture coefficients are sliced by the given ``subgroup`` and ``component``
        which means that if no subgroupd and/or component is given, multiple mixture
        coefficients are returned.

        If ``normalize`` is set to ``True``, the mixture coefficients are normalized
        along the component axis before being returned.
        """
        if getattr(self, "_mixture_coefs", None) is None:
            self._mixture_coefs = self._create_empty_mixture_coefs()

        if normalize:
            self.normalize_mixture_coefs()

        component = slice(None) if component is None else component
        subgroup = slice(None) if subgroup is None else subgroup
        return self._mixture_coefs.loc[component, subgroup]


    def set_mixture_coefs(
        self,
        new_mixture_coefs: float | np.ndarray,
        component: int | None = None,
        subgroup: str | None = None,
    ) -> None:
        """Assign new mixture coefficients to the model.

        As in :py:meth:`~get_mixture_coefs`, ``subgroup`` and ``component`` can be used
        to slice the mixture coefficients and therefore assign entirely new coefs to
        the entire model, to one subgroup, to one component, or to one component of one
        subgroup.
        """
        if getattr(self, "_mixture_coefs", None) is None:
            self._mixture_coefs = self._create_empty_mixture_coefs()

        component = slice(None) if component is None else component
        subgroup = slice(None) if subgroup is None else subgroup
        self._mixture_coefs.loc[component, subgroup] = new_mixture_coefs


    def normalize_mixture_coefs(self) -> None:
        """Normalize the mixture coefficients to sum to one."""
        if getattr(self, "_mixture_coefs", None) is not None:
            self._mixture_coefs = self._mixture_coefs / self._mixture_coefs.sum(axis=0)


    def repeat_mixture_coefs(self, t_stage: str, log: bool = True) -> np.ndarray:
        """Stretch the mixture coefficients to match the number of patients."""
        res = np.empty(shape=(0, len(self.components)))
        for label, subgroup in self.subgroups.items():
            num_patients = subgroup.diagnose_matrices[t_stage].shape[1]
            res = np.vstack([
                res,
                np.tile(self.get_mixture_coefs(subgroup=label), (num_patients, 1))
            ])

        return np.log(res) if log else res


    def get_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> Iterable[float] | dict[str, float]:
        """Get the parameters of the mixture model.

        This includes both the parameters of the individual components and the mixture
        coefficients. If a dictionary is returned (i.e. if ``as_dict`` is set to
        ``True``), the components' parameters are nested under keys that simply
        enumerate them. While the mixture coefficients are returned under keys of the
        form ``<subgroup>from<component>_coef``.

        The parameters are returned as a dictionary if ``as_dict`` is True, and as
        an iterable of floats otherwise. The argument ``as_flat`` determines whether
        the returned dict is flat or nested.

        >>> graph_dict = {
        ...     ("tumor", "T"): ["II", "III"],
        ...     ("lnl", "II"): ["III"],
        ...     ("lnl", "III"): [],
        ... }
        >>> mixture = LymphMixture(
        ...     model_kwargs={"graph_dict": graph_dict},
        ...     num_components=2,
        ... )
        >>> mixture.get_params(as_dict=True)     # doctest: +NORMALIZE_WHITESPACE
        {'0_TtoII_spread': 0.0,
         '0_TtoIII_spread': 0.0,
         '0_IItoIII_spread': 0.0,
         '1_TtoII_spread': 0.0,
         '1_TtoIII_spread': 0.0,
         '1_IItoIII_spread': 0.0}
        """
        params = {}
        for c, component in enumerate(self.components):
            params[str(c)] = component.get_params(as_flat=as_flat)

            for label in self.subgroups:
                params[str(c)].update({f"{label}_coef": self.get_mixture_coefs(c, label)})

        if as_flat or not as_dict:
            return flatten(params)

        return params if as_dict else params.values()


    def set_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Assign new params to the component models.

        This includes both the spread parameters for the component's models (if
        provided as positional arguments, they are used up first), as well as the
        mixture coefficients for the subgroups.

        Note:
            After setting all parameters, the mixture coefficients are normalized and
            may thus not be the same as the ones provided in the arguments.

        >>> graph_dict = {
        ...     ("tumor", "T"): ["II", "III"],
        ...     ("lnl", "II"): ["III"],
        ...     ("lnl", "III"): [],
        ... }
        >>> mixture = LymphMixture(
        ...     model_kwargs={"graph_dict": graph_dict},
        ...     num_components=2,
        ... )
        >>> mixture.set_params(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
        (0.7,)
        >>> mixture.get_params(as_dict=True)   # doctest: +NORMALIZE_WHITESPACE
        {'0_TtoII_spread': 0.1,
         '0_TtoIII_spread': 0.2,
         '0_IItoIII_spread': 0.3,
         '1_TtoII_spread': 0.4,
         '1_TtoIII_spread': 0.5,
         '1_IItoIII_spread': 0.6}
        """
        kwargs, global_kwargs = unflatten_and_split(
            kwargs, expected_keys=[str(c) for c, _ in enumerate(self.components)],
        )

        for c, component in enumerate(self.components):
            component_kwargs = global_kwargs.copy()
            component_kwargs.update(kwargs.get(str(c), {}))
            args = component.set_params(*args, **component_kwargs)

            for label in self.subgroups:
                first, args = popfirst(args)
                value = component_kwargs.get(f"{label}_coef", first)
                self.set_mixture_coefs(value, component=c, subgroup=label)

        self.normalize_mixture_coefs()
        return args


    def get_responsibilities(
        self,
        patient: int | None = None,
        subgroup: str | None = None,
        component: int | None = None,
        filter_by: tuple[str, str, str] | None = None,
        filter_value: Any | None = None,
    ) -> pd.DataFrame:
        """Get the repsonsibility of a ``patient`` for a ``component``.

        The ``patient`` index enumerates all patients in the mixture model unless
        ``subgroup`` is given, in which case the index runs over the patients in the
        given subgroup.

        Omitting ``component`` or ``patient`` (or both) will return corresponding slices
        of the responsibility table.
        """
        if subgroup is not None:
            resp_table = self.subgroups[subgroup].patient_data
        else:
            resp_table = self.patient_data

        if filter_by is not None and filter_value is not None:
            filter_idx = resp_table[filter_by] == filter_value
            resp_table = resp_table.loc[filter_idx]

        pat_slice = slice(None) if patient is None else patient
        comp_slice = (*RESP_COL, slice(None) if component is None else component)
        res = resp_table.loc[pat_slice,comp_slice]
        try:
            return res[RESP_COL]
        except (KeyError, IndexError):
            return res


    def set_responsibilities(
        self,
        new_responsibilities: float | np.ndarray,
        patient: int | None = None,
        subgroup: str | None = None,
        component: int | None = None,
    ):
        """Assign responsibilities to the model.

        They should have the shape ``(num_patients, num_components)`` and summing them
        along the last axis should yield a vector of ones.

        Note that these responsibilities essentially become the latent variables
        of the model if they are "hard", i.e. if they are either 0 or 1 and thus
        represent a one-hot encoding of the component assignments.
        """
        pat_slice = slice(None) if patient is None else patient
        comp_slice = (*RESP_COL, slice(None) if component is None else component)

        if subgroup is not None:
            sub_data = self.subgroups[subgroup].patient_data
            sub_data.loc[pat_slice,comp_slice] = new_responsibilities
            return

        patient_idx = 0
        for subgroup in self.subgroups.values():
            sub_data = subgroup.patient_data
            patient_idx += len(sub_data)

            if patient is not None:
                if patient_idx > patient:
                    sub_data.loc[pat_slice,comp_slice] = new_responsibilities
                    return

            else:
                sub_resp = new_responsibilities[:len(sub_data)]
                sub_data.loc[pat_slice,comp_slice] = sub_resp
                new_responsibilities = new_responsibilities[len(sub_data):]


    def normalize_responsibilities(self) -> None:
        """Normalize the responsibilities to sum to one."""
        for label in self.subgroups:
            sub_resps = self.get_responsibilities(subgroup=label)
            self.set_responsibilities(sub_resps / sub_resps.sum(axis=1), subgroup=label)


    def harden_responsibilities(self) -> None:
        """Make the responsibilities hard, i.e. convert them to one-hot encodings."""
        resps = self.get_responsibilities().to_numpy()
        max_resps = np.max(resps, axis=1)
        hard_resps = np.where(resps == max_resps[:,None], 1, 0)
        self.set_responsibilities(hard_resps)


    def load_patient_data(
        self,
        patient_data: pd.DataFrame,
        split_by: tuple[str, str, str],
        **kwargs,
    ):
        """Split the ``patient_data`` into subgroups and load it into the model.

        This amounts to computing the diagnose matrices for the individual subgroups.
        The ``split_by`` tuple should contain the three-level header of the LyProX-style
        data. Any additional keyword arguments are passed to the
        :py:meth:`~lymph.models.Unilateral.load_patient_data` method.
        """
        self._mixture_coefs = None
        grouped = patient_data.groupby(split_by)

        for label, data in grouped:
            if label not in self.subgroups:
                self.subgroups[label] = self._model_cls(**self._model_kwargs)
            data = join_with_responsibilities(
                data, num_components=len(self.components)
            )
            self.subgroups[label].load_patient_data(data, **kwargs)

        modalities.Composite.__init__(
            self,
            modality_children=self.subgroups,
            is_modality_leaf=False,
        )


    @property
    def patient_data(self) -> pd.DataFrame:
        """Return all patients stored in the individual subgroups."""
        return pd.concat([
            subgroup.patient_data for subgroup in self.subgroups.values()
        ], ignore_index=True)


    def comp_component_patient_likelihood(
        self,
        t_stage: str,
        log: bool = True,
    ) -> np.ndarray:
        """Compute the (log-)likelihood of all patients, given the components.

        The returned array has shape ``(num_components, num_patients)`` and contains
        the likelihood of each patient under each component. If ``log`` is set to
        ``True``, the likelihoods are returned in log-space.
        """
        stacked_diag_matrices = np.hstack([
            subgroup.diagnose_matrices[t_stage] for subgroup in self.subgroups.values()
        ])
        llhs = np.empty(shape=(stacked_diag_matrices.shape[1], len(self.components)))
        for i, component in enumerate(self.components):
            llhs[:,i] = component.comp_state_dist(t_stage=t_stage) @ stacked_diag_matrices

        return np.log(llhs) if log else llhs


    def comp_patient_mixture_likelihood(
        self,
        t_stage: str,
        log: bool = True,
        marginalize: bool = False,
    ) -> np.ndarray:
        """Compute the (log-)likelihood of all patients under the mixture model.

        This is essentially the (log-)likelihood of all patients given the individual
        components, but weighted by the mixture coefficients.

        If ``marginalize`` is set to ``True``, the likelihoods are summed
        over the components, effectively marginalizing the components out of the
        likelihoods.
        """
        component_patient_likelihood = self.comp_component_patient_likelihood(t_stage, log)
        full_mixture_coefs = self.repeat_mixture_coefs(t_stage, log)

        if log:
            llh = full_mixture_coefs + component_patient_likelihood
        else:
            llh = full_mixture_coefs * component_patient_likelihood

        if marginalize:
            return np.logaddexp.reduce(llh, axis=0) if log else np.sum(llh, axis=0)

        return llh


    def _incomplete_data_likelihood(
        self,
        t_stage: str | None = None,
        log: bool = True,
    ) -> float:
        """Compute the incomplete data likelihood of the model."""
        if t_stage is None:
            t_stages = self.t_stages
        else:
            t_stages = [t_stage]

        llh = 0 if log else 1.0
        for t in t_stages:
            llhs = self.comp_patient_mixture_likelihood(t, log, marginalize=True)

            if log:
                llh += np.sum(llhs)
            else:
                llh *= np.prod(llhs)

        return llh


    def _complete_data_likelihood(
        self,
        t_stage: str | None = None,
        log: bool = True,
    ) -> float:
        """Compute the complete data likelihood of the model."""
        if t_stage is None:
            t_stages = self.t_stages
        else:
            t_stages = [t_stage]

        llh = 0 if log else 1.0
        for t in t_stages:
            llhs = self.comp_patient_mixture_likelihood(t, log)
            resps = self.get_responsibilities(
                filter_by=T_STAGE_COL, filter_value=t
            ).to_numpy()

            if log:
                llh += np.sum(resps * llhs)
            else:
                llh *= np.prod(llhs ** resps)

        return llh


    def likelihood(
        self,
        given_params: Iterable[float] | dict[str, float] | None = None,
        log: bool = True,
        complete: bool = True,
    ) -> float:
        """Compute the (in-)complete data likelihood of the model.

        If ``complete`` is set to ``True``, the complete data likelihood is computed.
        Otherwise, the incomplete data likelihood is computed.

        The likelihood is computed for the ``given_params``. If no parameters are given,
        the likelihood is computed for the current parameters of the model.

        The likelihood is returned in log-space if ``log`` is set to ``True``.
        """
        try:
            # all functions and methods called here should raise a ValueError if the
            # given parameters are invalid...
            if given_params is None:
                pass
            elif isinstance(given_params, dict):
                self.set_params(**given_params)
            else:
                self.set_params(*given_params)
        except ValueError:
            return -np.inf if log else 0.

        if complete:
            return self._complete_data_likelihood(log=log)

        return self._incomplete_data_likelihood(log=log)


    def risk(self):
        raise NotImplementedError
