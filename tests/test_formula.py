import unittest

from formparse import Formula
from formparse.formula import (
    FormulaRuntimeError,
    FormulaSyntaxError,
    FormulaZeroDivisionError,
)

__author__ = "Nicklas Bocksberger"
__copyright__ = "Nicklas Bocksberger"
__license__ = "MIT"


class TestFormula(unittest.TestCase):
    """Test calculating with the `Formula` class.
    """

    def test_addition(self):
        """Test simple addition.
        """
        formula = Formula('x + 5')
        result = formula.eval({'x': 2})
        self.assertEqual(result, 7)

    def test_subtraction(self):
        """Test simple subtraction.
        """
        formula = Formula('x - 5')
        result = formula.eval({'x': 2})
        self.assertEqual(result, -3)

    def test_multiplication(self):
        """Test simple multiplication.
        """
        formula = Formula('x * 5')
        result = formula.eval({'x': 2})
        self.assertEqual(result, 10)

    def test_division(self):
        """Test simple division.
        """
        formula = Formula('x / 5')
        result = formula.eval({'x': 10})
        self.assertEqual(result, 2)

    def test_power(self):
        """Test simple power to the x.
        """
        formula = Formula('x**2')
        result = formula.eval({'x': 4})
        self.assertEqual(result, 16)

    def test_addition_with_negative(self):
        """Test simple addition with negative number.
        """
        formula = Formula('x + -5')
        result = formula.eval({'x': 2})
        self.assertEqual(result, -3)

    def test_subtraction_with_negative(self):
        """Test simple subtraction with negative number.
        """
        formula = Formula('x - -5')
        result = formula.eval({'x': 2})
        self.assertEqual(result, 7)

    def test_multiplication_with_negative(self):
        """Test simple multiplication with negative number.
        """
        formula = Formula('x * -5')
        result = formula.eval({'x': 2})
        self.assertEqual(result, -10)

    def test_division_with_negative(self):
        """Test simple division with negative number.
        """
        formula = Formula('x / -5')
        result = formula.eval({'x': 10})
        self.assertEqual(result, -2)

    def test_power_with_negative(self):
        """Test simple power with negative number.
        """
        formula = Formula('x ** -2')
        result = formula.eval({'x': 2})
        self.assertEqual(result, 0.25)

    def test_without_args(self):
        """Test constant expression.
        """
        formula = Formula('2*3')
        result = formula.eval()
        self.assertEqual(result, 6)

    def test_with_additional_args(self):
        """Test passing more args than needed.
        """
        formula = Formula('2*3')
        result = formula.eval({'x': 4})
        self.assertEqual(result, 6)

    @unittest.skip('Not implemented yet')
    def test_abort_for_big_numbers(self):
        """Test aborting the evaluation if the results might become too big.
        """

    def test_faisl_for_division_by_zero(self):
        """Test failing for division by zero.
        """
        formula = Formula('3/x')
        self.assertRaises(FormulaZeroDivisionError, lambda: formula.eval({'x': 0}))

    @unittest.skip('Not implemented yet')
    def test_fails_for_root_of_negative(self):
        """Test failing for calculating root of negative numbers.
        """
        formula = Formula('(-2)**0.5')
        self.assertRaises(FormulaRuntimeError, formula.eval)

    def test_fails_for_missing_arg(self):
        """Test failing for missing argument.
        """
        formula = Formula('x + y')
        self.assertRaises(FormulaRuntimeError, lambda: formula.eval({'x': 5}))
        formula = Formula('x + y')
        self.assertRaises(FormulaRuntimeError, lambda: formula.eval({'x': 5, 'z': 6}))

    def test_fails_for_invalid_formula(self):
        """Test fails for invalid formula with not supported operator.
        """
        self.assertRaises(FormulaSyntaxError, lambda: Formula('x//5').eval({'x': 2}))
        # self.assertRaises(FormulaSyntaxError, lambda: Formula('x**5').eval({'x': 2}))

    def test_fails_for_invalid_args(self):
        """Test fails for invalid argument type.
        """
        formula = Formula('x*4')
        self.assertRaises(FormulaRuntimeError, lambda: formula.eval('x'))

    def test_addition_with_float(self):
        """Test simple addition with float.
        """
        formula = Formula('2.5+x')
        result = formula.eval({'x': 3.75})
        self.assertEqual(result, 6.25)

    def test_subtraction_with_float(self):
        """Test simple subtraction with float.
        """
        formula = Formula('2.5-x')
        result = formula.eval({'x': 3.75})
        self.assertEqual(result, -1.25)

    def test_multiplication_with_float(self):
        """Test simple multiplication with float.
        """
        formula = Formula('2.5*x')
        result = formula.eval({'x': 3})
        self.assertEqual(result, 7.5)

    def test_division_with_float(self):
        """Test simple division with float.
        """
        formula = Formula('2.5/x')
        result = formula.eval({'x': 1.25})
        self.assertEqual(result, 2)


if __name__=='__main__':
    unittest.main()
