"""Make doctests of the lymixture package discoverable by unittest."""

import doctest
import unittest
import warnings

import pandas as pd

from lymixture import models, utils


def load_tests(loader, tests: unittest.TestSuite, ignore):
    """Load the doctests of the lymixture package."""
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    tests.addTests(doctest.DocTestSuite(models))
    tests.addTests(doctest.DocTestSuite(utils))
    return tests
