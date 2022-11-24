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

    MAX_FORMULA_LENGTH = 255

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
        return False, f'Unsopported Function {node}'

    def eval(self, args: Optional[dict]={}) -> float:
        """Evaluate the formula for a set if given arguments

        Args:
            args (Optional[dict], optional): A dictionary with the arguments. Defaults to {}.

        Raises:
            FormulaRuntimeError: If the arguments are not a dictionary.
            FormulaRuntimeError: If the evaluation fails for any other reason.

        Returns:
            float: The value of the result.
        """
        if not isinstance(args, dict):
            raise FormulaRuntimeError(
                f'Invalid type `{type(args)}` for args, only `dict` supported.')
        try:
            return self._eval_node(self.formula, self.node, args)
        except FormulaSyntaxError:
            raise
        except ZeroDivisionError:
            raise FormulaZeroDivisionError from ZeroDivisionError
        except Exception as exception:
            raise FormulaRuntimeError(f'Evaluation failed: {exception}') from exception

    def _eval_node(self, source: str, node: ast.AST, args: Dict[str, Any]) -> float:
        for ast_type, eval_name in self.EVALUATORS.items():
            if isinstance(node, ast_type):
                evaluator = getattr(self, eval_name)
                return evaluator(source, node, args)
        raise FormulaSyntaxError('Could not evaluate, might be due to unsupported operator.')

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
