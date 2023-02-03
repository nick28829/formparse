"""Formula that can be parsed and evaluated
"""
import ast
import logging
import operator
from typing import Any, Dict, Optional, Tuple

__author__ = 'Nicklas Bocksberger'
__copyright__ = 'Nicklas Bocksberger'
__license__ = 'MIT'

_logger = logging.getLogger(__name__)


class FormulaException(Exception):
    """Generic Exception for `Formula`, base class for `FormulaSyntaxError`
    and `FormulaRuntimeError`.
    """

class FormulaSyntaxError(FormulaException):
    """Exception raised if there is an error in the syntax of the formula input.
    """

class FormulaRuntimeError(FormulaException):
    """Exception raised if there is an error during the runtime of the formula,
    especially with the argument input.
    """

class FormulaZeroDivisionError(FormulaRuntimeError):
    """Exception raised if there is a division throgh 0 error.
    """

class FormulaComplexityError(FormulaRuntimeError):
    """Exception raised if the expected result is to big to calculate.
    """

class Formula:
    """Simple formula, generated from a string input can it be evaluated with it
    `.eval()`method. The currently supported operators are `+`, `-`, `*` and `/`.
    """

    EVALUATORS = {
        ast.Expression: '_eval_expression',
        ast.Constant: '_eval_constant',
        ast.Name: '_eval_name',
        ast.BinOp: '_eval_binop',
        ast.UnaryOp: '_eval_unaryop',
        ast.Call: '_eval_call',
    }

    BIN_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }

    UN_OPERATORS = {
        ast.USub: operator.neg,
    }

    FUNCTIONS = {
        'max': max,
        'abs': abs,
        'min': min,
    }

    MAX_FORMULA_LENGTH = 255
    """Maximum length accepted for the formula, as is (incl. whitespaces, etc.)
    """

    MAX_RESULT_POTENCY = 18
    """Maximum maximum power of ten that is allowed for a expected result.
    """

    def __init__(self, formula: str) -> None:
        """
        Args:
            formula (str): The formula as a string, arguments can passed during
            evaluation.

        Raises:
            FormulaSyntaxError: Raised if the formula is not valid.
        """
        self.formula = formula
        self.node = self.parse_formula(self.formula)
        self._validate_formula()

    def _validate_formula(self):
        if self.MAX_FORMULA_LENGTH and len(self.formula) > self.MAX_FORMULA_LENGTH:
            raise FormulaSyntaxError('Formula can be 255 characters maximum.')
        valid, *problem = self.validate(self.node)
        if not valid:
            raise FormulaSyntaxError(problem)

    @classmethod
    def parse_formula(cls, formula: str) -> ast.AST:
        """Parse a given formula into an `ast` node.

        Args:
            formula (str): Formula to parse.

        Returns:
            ast.AST: Parsed node.
        """
        try:
            return ast.parse(formula, '<string>', mode='eval')
        except SyntaxError as exception:
            raise FormulaSyntaxError('Could not parse formula.') from exception

    @classmethod
    def validate(cls, node: ast.AST or str) -> Tuple[bool, str or None]:
        """Check whether or not formula provided in `node` is valid.

        `NOTE`: `Formula.validate()` does not check for length constraint
        but just for general validity.
        Args:
            node (ast.AST | str): Formula to check, either as `str` or parsed `ast` Tree.

        Returns:
            Tuple[bool, str | None]: If the formula is valid, if not,
            provide reason in second value.
        """
        if isinstance(node, str):
            try:
                node = cls.parse_formula(node)
            except FormulaSyntaxError as exception:
                return False, str(exception)
        # TODO: change to case match in version 1, change or to union operator
        if isinstance(node, ast.Expression):
            if type(node) in cls.EVALUATORS:
                return cls.validate(node.body)
            return False, 'Unknown function.'
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, int) or isinstance(node.value, float):
                return True, None
            return False, f'Unsopported constant type {type(node.value)}'
        elif isinstance(node, ast.Name):
            return True, None
        elif isinstance(node, ast.BinOp):
            if type(node.op) in cls.BIN_OPERATORS:
                return cls.validate(node.left) and cls.validate(node.right)
            return False, f'Unsopported operator {node.op}'
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) in cls.UN_OPERATORS:
                return cls.validate(node.operand)
            return False, f'Unsopported operator {node.op}'
        elif isinstance(node, ast.Call):
            if node.func.id in cls.FUNCTIONS:
                return all(cls.validate(arg) for arg in node.args), None
            return False, f'Unsupported function {node.func.id}'
        return False, f'Unsopported Function {node}'

    def eval(self, args: Optional[dict]=None) -> float:
        """Evaluate the formula for a set if given arguments

        Args:
            args (Optional[dict], optional): A dictionary with the arguments. Defaults to {}.

        Raises:
            FormulaRuntimeError: If the arguments are not a dictionary.
            FormulaRuntimeError: If the evaluation fails for any other reason.

        Returns:
            float: The value of the result.
        """
        if args and not isinstance(args, dict):
            raise FormulaRuntimeError(
                f'Invalid type `{type(args)}` for args, only `dict` supported.')
        try:
            self.validate_result_size(args)
        except FormulaComplexityError as exception:
            raise exception
        try:
            return self._eval_node(self.formula, self.node, args)
        except FormulaSyntaxError:
            raise
        except ZeroDivisionError:
            raise FormulaZeroDivisionError from ZeroDivisionError
        except Exception as exception:
            raise FormulaRuntimeError(f'Evaluation failed: {exception}') from exception

    def _eval_call(self, source: str, node: ast.AST, args: Dict[str, Any]) -> float:
        try:
            func = self.FUNCTIONS[node.func.id]
        except KeyError as exception:
            raise FormulaSyntaxError(f'Function {node.func.id} not supported') from exception
        return func(*[self._eval_node(source, arg, args) for arg in node.args])

    def _eval_node(self, source: str, node: ast.AST, args: Dict[str, Any]) -> float:
        try:
            eval_name = self.EVALUATORS[type(node)]
        except KeyError as exception:
            raise FormulaSyntaxError(
                'Could not evaluate, might be due to unsupported operator.') from exception
        evaluator = getattr(self, eval_name)
        return evaluator(source, node, args)

    def _eval_expression(self, source: str, node: ast.Expression, args: Dict[str, Any]) -> float:
        return self._eval_node(source, node.body, args)

    def _eval_constant(self, _: str, node: ast.Constant, __: Dict[str, Any]) -> float:
        if isinstance(node.value, int) or isinstance(node.value, float):
            return float(node.value)
        else:
            raise FormulaSyntaxError(f'Unsupported type of constant {node.value}.')

    def _eval_name(self, _: str, node: ast.Name, args: Dict[str, Any]) -> float:
        try:
            return float(args[node.id])
        except KeyError as exception:
            raise FormulaRuntimeError(f'Undefined variable: {node.id}') from exception

    def _eval_binop(self, source: str, node: ast.BinOp, args: Dict[str, Any]) -> float:
        left_value = self._eval_node(source, node.left, args)
        right_value = self._eval_node(source, node.right, args)

        try:
            evaluator = self.BIN_OPERATORS[type(node.op)]
        except KeyError as exception:
            raise FormulaSyntaxError('Operations of this type are not supported') from exception

        return evaluator(left_value, right_value)

    def _eval_unaryop(self, source: str, node: ast.UnaryOp, args: Dict[str, Any]) -> float:
        operand_value = self._eval_node(source, node.operand, args)

        try:
            apply = self.UN_OPERATORS[type(node.op)]
        except KeyError as exception:
            raise FormulaSyntaxError('Operations of this type are not supported') from exception

        return apply(operand_value)

    def __str__(self) -> str:
        return f'<formparse.Formula {self.formula[:32]}>'

    def validate_result_size(self, args: Optional[dict]=None) -> None:
        """Make sure that the maximum estimated result is not bigger
        than as set by `MAX_RESULT_POTENCY`.

        Args:
            args (Optional[dict], optional): Parameters provided for
            evaluation, if not provided, each variable is assigned
            potency of 1. Defaults to None.

        Raises:
            FormulaComplexityError: If the expected result is bigger than allowed.
        """
        result_potency = self.estimate_result_size(self.node, args)
        if result_potency > self.MAX_RESULT_POTENCY:
            raise FormulaComplexityError(
                f'Expected result with maximum size 10^{result_potency} is too big.')

    @classmethod
    def estimate_result_size(cls, node, args: Optional[dict]=None) -> int:
        """Estimate the result size based one the power of tens of its operands.

        A more indepth explanation will follow.

        Args:
            node (ast.AST): Node to estimate result size for.
            args (Optional[dict], optional): Parameters provided for
            evaluation, if not provided, each variable is assigned
            potency of 1. Defaults to None.

        Raises:
            FormulaComplexityError: If calculating the result size itself
            is to expensive.

        Returns:
            int: The maximum power to ten of the result.
        """
        def potency(x: float):
            return len(str(int(x))) - 1 # might not be elegant but it's simple

        if isinstance(node, ast.Expression):
            return cls.estimate_result_size(node.body, args)
        if isinstance(node, ast.Constant):
            return potency(node.value)
        if isinstance(node, ast.Name):
            if args and node.id in args:
                return potency(args[node.id])
            _logger.warning('Calculating complexity of Formula with unknown %s.', node.id)
            return 1
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return max(cls.estimate_result_size(node.left, args),
                           cls.estimate_result_size(node.right, args),
                        ) + 2
            if isinstance(node.op, ast.Mult):
                return cls.estimate_result_size(node.left, args) \
                        + cls.estimate_result_size(node.right, args) \
                        + 2
            if isinstance(node.op, ast.Div):
                pot_l = cls.estimate_result_size(node.left, args)
                pot_r = cls.estimate_result_size(node.right, args)
                return int((( pot_l + 1) / (pot_r + 1 )) + 0.5)
            if isinstance(node.op, ast.Pow):
                pot_l = cls.estimate_result_size(node.left, args) # biggest potency possible on left
                pot_r = cls.estimate_result_size(node.right, args) # "" on right
                if pot_r > 5:
                    raise FormulaComplexityError(
                            'Can\'t calculate result size due to the partial result size')
                if pot_r > 3:
                    a = cls.estimate_result_size(cls.parse_formula(f'10**({pot_r} + 1) - 1'))
                    if a < cls.MAX_RESULT_POTENCY:
                        raise FormulaComplexityError(
                            'Can\'t calculate result size due to the partial result size')
                max_val_r = 10 ** (pot_r + 1) - 1 # biggest value possible on right site
                return (pot_l + 1) * max_val_r
        if isinstance(node, ast.UnaryOp):
            return cls.estimate_result_size(node.operand)
        if isinstance(node, ast.Call):
            try:
                return max(cls.estimate_result_size(arg) for arg in node.args)
            except ValueError:
                _logger.warning(
                    'Calculating complexity of Formula with function %s with unknown result size.',
                    node.func.id,
                )
                return 1
