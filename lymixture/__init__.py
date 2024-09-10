"""
This package defines classes and functions to model lymphatic involvement as a
mixture of hidden Markov models as hey are implemented in the `lymph-model` package.
"""

import logging

from lymixture._version import version
from lymixture.models import LymphMixture

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["LymphMixture"]
__version__ = version
__author__ = "Roman Ludwig"
__email__ = "roman.ludwig@usz.ch"
__uri__ = "https://github.com/rmnldwg/lymixture"
