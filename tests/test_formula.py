import unittest

from formparse import Formula
from formparse.formula import (
    FormulaComplexityError,
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

    def test_fails_for_division_by_zero(self):
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

    def test_addition_with_constant_formula(self):
        """Test simple addition with no extra argument.
        """
        formula = Formula('2 + 2')
        result = formula.eval()
        self.assertEqual(result, 4)

    def test_subtraction_with_constant_formula(self):
        """Test simple subtraction with no extra argument.
        """
        formula = Formula('2 - 2')
        result = formula.eval()
        self.assertEqual(result, 0)

    def test_multiplication_with_constant_formula(self):
        """Test simple multiplication with no extra argument.
        """
        formula = Formula('2 * 2')
        result = formula.eval()
        self.assertEqual(result, 4)

    def test_division_with_constant_formula(self):
        """Test simple division with no extra argument.
        """
        formula = Formula('2 / 2')
        result = formula.eval()
        self.assertEqual(result, 1)


class TestFormulaValidation(unittest.TestCase):
    """Test for `Formula` validation methods.
    """

    def test_validate_works_for_string(self):
        """Test validation works with `Formula.validate()` and a `str` argument.
        """
        valid, _ = Formula.validate('3*(a+7)')
        self.assertTrue(valid)
        valid, _ = Formula.validate('3)')
        self.assertFalse(valid)

    def test_validate_works_for_node(self):
        """Test validation works with `Formula.validate()` and a `ast.AST` argument.
        """
        import ast
        valid, _ = Formula.validate(ast.parse('4*5', '<string>', mode='eval'))
        self.assertTrue(valid)
        valid, _ = Formula.validate(ast.parse('3//7', '<string>', mode='eval'))
        self.assertFalse(valid)

    def test_fails_for_invalid_operator(self):
        """Test fails for creating formula with invalid operator.
        """
        self.assertRaises(FormulaSyntaxError, lambda: Formula('a // 5'))

    def test_succeeds_for_valid_formula(self):
        """Test succeeds for valid formula.
        """
        Formula('5*(4/7)**6*x')
        self.assertTrue(True)

    def test_fails_for_length_constraint(self):
        """Test fails for string longer than 255 characters.
        """
        f = '2' + 100 * '+ 2'
        self.assertRaises(FormulaSyntaxError, lambda: Formula(f))

    def test_succeeds_for_validate_long_formula(self):
        """Test succeeds with `validate()` method since it does not check length constraint.
        """
        f = '2' + 100 * '+ 2'
        Formula.validate(f)
        self.assertTrue(True)


class TestFormulaOperationEstimation(unittest.TestCase):
    """Test for `Formula`operation size estimation.
    """

    def test_fails_for_complex_calculation(self):
        """Test fails for complex calculation.
        """
        formula = Formula('99**9999**9999')
        self.assertRaises(FormulaComplexityError, formula.eval)

    def test_estimate_addition_size(self):
        """Estimation is correct for simple addition.
        """
        formula = Formula('99+9999')
        result = formula.eval()
        size_estimation = Formula.estimate_result_size(formula.node)
        self.assertEqual(len(str(int(result))), size_estimation)

    def test_estimate_subtraction_size(self):
        """Estimation is correct for simple subtraction.
        """
        formula = Formula('-99-9999')
        result = formula.eval()
        size_estimation = Formula.estimate_result_size(formula.node)
        # subtract 1 for negative sign
        self.assertEqual(len(str(int(result))) - 1, size_estimation)

    def test_estimate_multiplicationn_size(self):
        """Estimation is correct for simple multiplication.
        """
        formula = Formula('99*9999')
        result = formula.eval()
        size_estimation = Formula.estimate_result_size(formula.node)
        self.assertEqual(len(str(int(result))), size_estimation)

    def test_estimate_division_size(self):
        """Estimation is correct for simple division.
        """
        formula = Formula('1000/99')
        result = formula.eval()
        size_estimation = Formula.estimate_result_size(formula.node)
        self.assertEqual(len(str(int(result))), size_estimation)

    def test_estimate_power_size(self):
        """Estimation is correct for simple power.
        """
        formula = Formula('99**9')
        result = formula.eval()
        size_estimation = Formula.estimate_result_size(formula.node)
        self.assertEqual(len(str(int(result))), size_estimation)


class TestFormulaFunctions(unittest.TestCase):
    """Test for functions that can be used inside `Formula`.
    """

    def test_max_function(self):
        """Test `max` function for correctness.
        """
        formula = Formula('max(1, x)')
        result = formula.eval({'x': 2})
        self.assertEqual(result, 2)
        result = formula.eval({'x': 0})
        self.assertEqual(result, 1)

    def test_min_function(self):
        """Test `min` function for correctness.
        """
        formula = Formula('min(1, x)')
        result = formula.eval({'x': 2})
        self.assertEqual(result, 1)
        result = formula.eval({'x': 0})
        self.assertEqual(result, 0)

    def test_abs_function(self):
        """Test `abs` function for correctness.
        """
        formula = Formula('abs(x)')
        result = formula.eval({'x': -5})
        self.assertEqual(result, 5)
        result = formula.eval({'x': 5})
        self.assertEqual(result, 5)

    def test_fail_abs_without_args(self):
        """Test fails for `abs` without arguments.
        """
        formula = Formula('abs()')
        self.assertRaises(FormulaRuntimeError, formula.eval)

    def test_fail_min_without_args(self):
        """Test fails for `min` without arguments.
        """
        formula = Formula('min()')
        self.assertRaises(FormulaRuntimeError, formula.eval)

    def test_fail_max_without_args(self):
        """Test fails for `max` without arguments.
        """
        formula = Formula('max()')
        self.assertRaises(FormulaRuntimeError, formula.eval)

    def test_fail_abs_with_multiple_args(self):
        """Test fails for `abs` with multiple arguments.
        """
        formula = Formula('abs(1, 2)')
        self.assertRaises(FormulaRuntimeError, formula.eval)


if __name__=='__main__':
    unittest.main()
