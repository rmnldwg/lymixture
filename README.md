<img src="https://raw.githubusercontent.com/rmnldwg/lymixture/dev/github-social-card.png" alt="social card" style="width:830px;"/>

## What is `lyMixture`?

This package is an extension to the [`lymph`] package, which models lymphatic tumor progression in head and neck cancer.

However, different tumor locations/subsites may exhibit different spread patterns. This extension to the original model attempts to model this as a mixture of several components that might represent something like the atomic spread patterns.

This code was originally written by [Julian Brönnimann] as part of his Master's thesis in the medical physics research group of Prof. Jan Unkelbach at the University Hospital Zurich. It was subsequently adapted and extended to match the [`lymph`] API as closely as possible.

[`lymph`]: https://lymph-model.readthedocs.io
[Julian Brönnimann]: https://github.com/julianbro

## Installation

At the moment, you need to install the package from source. To do so, first clone the repository and `cd` into it:

```
git clone https://github.com/rmnldwg/lymixture
cd lymixture
```

Then, preferably inside a virtual environment, like [`venv`], install the package via

```
pip install .
```

[`venv`]: https://docs.python.org/3.10/library/venv.html

## Documentation

The docs for this package's API can be found on <https://lymixture.readthedocs.io> (not yet).
