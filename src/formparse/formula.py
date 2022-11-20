"""Formula that can be parsed and evaluated
"""
import ast
import logging
import operator
from typing import Any, Dict, Optional

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


    def __init__(self, formula: str) -> None:
        """
        Args:
            formula (str): The formula as a string, arguments can passed during
            evaluation.

        Raises:
            FormulaSyntaxError: Raised if the formula is not valid.
        """
        self.formula = formula
        try:
            self.node = ast.parse(formula, '<string>', mode='eval')
        except SyntaxError as exception:
            raise FormulaSyntaxError('Could not parse formula.') from exception

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
