.. image:: https://readthedocs.org/projects/formparse/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://formparse.readthedocs.io/en/latest/

.. image:: https://img.shields.io/pypi/v/formparse.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/formparse/

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=========
formparse
=========


    Simple library for evaluating mathematical formulas.


Written as an safe alternative to Pythons ``eval()`` function the aim was to provide a lightweight library that could
evaluate mathematical formulas provided by users in a safe way.

.. _installation:

Installation
============
You can install this package unsing pip:

.. code-block:: bash

    pip install formparse


SECURITY WARNING!
-----------------
.. note::

    This package is currently pre-stable and some security features are still missing.


.. _usage:

Usage
=====
.. code-block:: python

    from formparse import Formula

    formula = Formula('3*x**2')

    result = formula.eval({'x': 2})


.. _alternatives:

Alternatives
============

``numexpr``:
------------

``numexpr`` is a Python library for ``numpy`` that accelerates array operations
but it can hadnle simple string calculations as well. It is therefore a lot bigger than
``formparse`` but if you are using ``numpy`` anyways in you project or need additional
features that we do not provide this might be a good fit. Check out its repo 
`here <https://github.com/pydata/numexpr>`_ .
