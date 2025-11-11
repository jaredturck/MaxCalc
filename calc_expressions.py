''' Module expressions '''

import calc_settings
from calc_operators import Operator, Prefix, Postfix, Infix, Ternary
import calc_op as op
from calc_vars import Value, Var, WordToken, LValue
from calc_errors import CalculatorError, ParseError
from calc_functions import Function, LFunc

st = calc_settings.Settings()
dummy = Var('dummy')

class Expression(Value):
    ''' Expression class representing a mathematical expression '''

    def __init__(self, input_str=None, brackets='', offset=0):
        self.input_str = input_str  # only the part of the string relevant to this Expression.
        self.tokens = []
        self.token_pos = []  # position of individual tokens within this Expression.
        self.parsed = None
        self.parsed_pos = None
        self.brackets = brackets  # 2-char string, '()', '[]', '{}' or an empty string.
        self.offset = offset  # position of this Expression relative to input string.

    def value(self, mem, debug=False):
        ''' Evaluate the expression and return its value '''
        from calc_tuples import Tuple, LTuple

        def evaluate(power=0, index=0, skip_eval=False):  # returns (Value, endIndex)
            def try_operate(l, *args, **kwargs):
                try:
                    ret = dummy if skip_eval else token.function(l, *args, **kwargs)
                    if debug:
                        print(f"{str(token).strip(): ^3}:", ', '.join((str(l),) + tuple(str(a) for a in args)))
                    return ret
                except CalculatorError as e:
                    raise (type(e))(e.args[0], token_pos)

            l = None
            while True:
                token = self.parsed[index]
                token_pos = self.parsed_pos[index]
                if isinstance(token, WordToken) and not skip_eval:
                    try:
                        split_list, var_list = token.splitWordToken(mem, self.parsed[index+1])
                    except CalculatorError as e:
                        raise (type(e))(e.args[0], self.parsed_pos[index])
                    self.parsed[index:index+1] = var_list
                    prev = 0
                    self.parsed_pos[index:index+1] = [(self.parsed_pos[index][0] + prev, self.parsed_pos[index][0] + (prev := prev + len(s))) for s in ([''] + split_list)[:-1]]
                    continue
                match l, token:
                    case None, Value() | WordToken():
                        l = dummy if skip_eval else token.value(mem=mem)
                        index += 1
                        continue
                    # case None, Postfix() | Infix():
                    #     raise ParseError(f"Unexpected operator '{token.name}'", self.parsed_pos[index])
                    # non-Fn-Fn : Low
                    # non-Fn-Prefix : Low
                    # non-Fn-non-Fn : High
                    # Fn-Fn : High
                    case Function(), Expression():
                        if not isinstance(token, Tuple):
                            self.parsed[index] = Tuple.fromExpr(token)
                        token = op.functionInvocation
                    case Value(), Function() | Prefix() if not isinstance(l, Function):  # Fn-Fn = High, nonFn-Fn = Low
                        # implicit mult of value to function / prefix, slightly lower precedence. For cases like 'sin2xsin3y'
                        # also need to handle stuff like 1/2()
                        token = op.implicitMultPrefix
                    case Value(), Value() | Expression():
                        token = op.implicitMult
                match token:
                    case Operator() if token.power[0] <= power: return l, index - 1
                    case Prefix():
                        l, index = evaluate(power=token.power[1], index=index+1, skip_eval=skip_eval)
                        l = try_operate(l)
                    case Ternary():
                        if token == op.ternary_else:
                            raise ParseError("Unexpected operator ' : '", self.parsed_pos[index])
                        from calc_number import zero
                        ternary_index = index
                        is_true = None
                        if not skip_eval:
                            is_true = op.eq.function(l, zero) == zero
                        true_val, index = evaluate(power=token.power[1], index=index+1, skip_eval=skip_eval or not is_true)
                        if self.parsed[index + 1] != op.ternary_else:
                            raise ParseError("Missing else clause ':' for ternary operator", self.parsed_pos[ternary_index])
                        false_val, index = evaluate(power=op.ternary_else.power[1], index=index+2, skip_eval=skip_eval or is_true)
                        if not skip_eval:
                            l = true_val if is_true else false_val
                    case Postfix():
                        l = try_operate(l)
                    case op.assignment | op.lambdaArrow:
                        if not isinstance(l, LValue) and not skip_eval:
                            raise ParseError(f"Invalid LValue for operator '{token.name}'", self.parsed_pos[index - 1])
                        old_index = index
                        if isinstance(l, LFunc) or token == op.lambdaArrow:  # create a function
                            closure = mem.copy()
                            if token == op.lambdaArrow:
                                func_name = None
                                if not isinstance(l, LTuple):  # build a tuple with this
                                    inner_expr = self.morphCopy(Expression)
                                    inner_expr.brackets = ''
                                    inner_expr.tokens = self.parsed[index - 1: index]
                                    inner_expr.token_pos = self.parsed_pos[index - 1: index]
                                    func_params = LTuple(inner_expr)
                                else:
                                    func_params = l
                            else:
                                func_name, func_params = l.name, l.params
                            if isinstance(closure_expression := self.parsed[index + 1], Closure):
                                old_index += 1
                                index += 1
                                closure_expression.value(closure)  # populate closure with the expressions inside it.
                            _, index = evaluate(power=token.power[1], index = index + 1, skip_eval=True)
                            expr = self.morphCopy()
                            expr.brackets = ''
                            expr.tokens = self.parsed[old_index + 1: index + 1]
                            expr.token_pos = self.parsed_pos[old_index + 1: index + 1]
                            expr.input_str = expr.input_str[expr.token_pos[0][0]: expr.token_pos[-1][1]]
                            to_assign = Function(func_name, func_params, expr, closure)
                        elif isinstance(l, LValue):
                            to_assign, index = evaluate(power=token.power[1], index = index + 1, skip_eval=skip_eval)
                        else: to_assign = None
                        l = try_operate(l, to_assign, mem=mem)
                        # f(x) = g(y) = x + y
                        # handle LFunc and WordTokens differently.
                        # - LFunc: save the following Expression without evaluating it.
                        # - WordToken: save the VALUE of the following Expression.
                    case Infix():
                        from calc_number import zero, one
                        old_index = index
                        exp, index = evaluate(power=token.power[1], index = index + 1 - (token in [op.implicitMult, op.implicitMultPrefix, op.functionInvocation]), skip_eval = skip_eval or token == op.logicalAND and op.eq.function(l, zero) == one or token == op.logicalOR and op.eq.function(l, zero) == zero)
                        if isinstance(exp, LValue):
                            raise ParseError("Invalid operation on LValue", self.parsed_pos[old_index])
                        l = try_operate(l, exp)
                    case None:
                        return l, index - 1
                index += 1

        self.parsed = self.tokens + [None, None]
        self.parsed_pos = self.token_pos + [(9999, 9999), (9999, 9999)]
        return evaluate()[0]


    def __str__(self):
        # from number import RealNumber
        s = ''
        for token in self.tokens:
            s += token.fromString if hasattr(token, 'fromString') else str(token)
        if self.brackets:
            s = self.brackets[0] + s + self.brackets[-1]
        return s

    def __repr__(self):
        return f"Expression('{str(self)}')"


class Closure(Expression):
    ''' Closure class representing a closure expression '''
    def __repr__(self):
        return f"Closure('{str(self)}')"
