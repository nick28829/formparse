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

Additionally, there the functions ``min()``, ``max()`` and ``abs()`` are available.
They can be used the same way as in Python.


.. _evaluating_a_formula:

Evaluating a Formula
====================

For evaluating the formula you simply call the ``.eval()`` method, passing
it a dictionary with the variables you want to provide the formula with.

.. code-block:: python


    result = formula.eval({'x': 2})


.. _validation:

Validation:
===========
When calling the contructor, a validator is automatically called. It ensures
that the formula is parseable, is not longer than the length constraint and
all the operators are supported. If there is an error, a ``FormulaSyntaxError``
is raised. You can manually validate a formula using the ``.validate()`` class
method. 

.. code-block:: python


    valid, problem = Formula.validate('a^2')
    if not valid:
        print(problem) # outputs 'Unsopported operator'


The function returns if the formula is valid and, if it is not, the reason why
it is invalid.

.. _complexity_check:

Complexity Check:
=================
Before acutally evaluating a formula with the ``eval()`` method, a complexity check
is run automatically. This is intended to prevent user from crashing the program
complex calculations. The check is based on the power of tens of the operands. If
the maximum possible result is bigger than 10^18, a ``FormulaComplexityError``
is thrown. For examle:

.. code-block:: python


    Formula('65**500').eval() # throws `FormulaComplexityError` error


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
