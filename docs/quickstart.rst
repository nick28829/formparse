=========
Qickstart
=========

.. _creating_a_formula:

Creating a Formula
==================

The usage of the package is quite simple. For creating a formula, you simply
import the ``Formula`` class and call its constructor with the formula as
the argument:

.. code-block:: python

    from formparse import Formula

    formula = Formula('3*x**2')


Since the package uses ``ast`` for building the syntax tree the syntax and
operators are those used in Python. Concretely, the currently available
operators are ``+``, ``-``, ``*``, ``/`` and ``**``.


.. _evaluating_a_formula:

Evaluating a Formula
====================

For evaluating the formula you simply call the ``.eval()`` method, passing
it a dictionary with the variables you want to provide the formula with.

.. code-block:: python


    result = formula.eval({'x': 2})


.. _limitations:

Limitations
===========

Currently, there is not support for currying/partially applying the formula.
If you don't pass in all arguments, a ``formula.FormulaRuntimeError`` will be thrown.

Also, currently there is no support for positinal arguments. If you pass anything
different than a dictionary as argument, a ``formula.FormulaRuntimeError`` will be
thrown.

It is possible to evaluate a formula that does not contain any variables. Also,
passing additional variables in the dictionary does not cause an error. Therefore,
you can use ``Formula`` also for constant values if needed.
