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


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
