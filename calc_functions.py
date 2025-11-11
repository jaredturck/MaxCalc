''' Module functions '''

from calc_vars import Value, LValue
from calc_errors import ParseError, EvaluationError

# Functions must be followed by bracketed expressions, unlike Unary_Left_Operators.
# Trig fns and sqrt are therefore treated as Unary_Left_Operators.
# ffg(7), f(g(3, 4)), (fg^2)(x), (f(fg)^2)(7)

# LAMBDAS
# sum = x => y => z => x + y + z
# sum(3): 'y => {x = 3}; z => x + y + z'
# sum(3, 5): 'z => {x = 3; y = 5}; x + y + z'

class Function(Value):
    ''' Function class representing a mathematical function '''

    def __init__(self, name='<fn>', params=None, expr=None, closure=None):
        self.name = name
        self.function = self.invoke
        self.func_list = [self]
        self.expression = expr
        if closure is None:
            raise MemoryError("Function must be created with a closure. Pass the global memory mainMem if it is defined in global scope.")
        self.closure = closure
        if self.name is not None:
            self.closure.add(self.name, self)  # Add self to closure
        if params is None:
            raise ParseError("Function parameter should be exactly one LTuple")
        self.params = params

    def value(self, *args, **kwargs):
        return self

    def __str__(self):
        closure_str = self.closure.str_without(self.name) + ' ' if len(self.closure.vars) > (self.name is not None) else ''
        if self.name is None:  # lambdas
            if len(self.params) == 1 and len(self.params.tokens[0].tokens) == 1:
                first_param = str(self.params)[1:-1]
            else:
                first_param = str(self.params)
            return f"{first_param} => {closure_str}{self.expression}"
        if hasattr(self, 'params') and hasattr(self, 'expression'):
            return f"{self.name}{self.params} = {closure_str}{self.expression}"
        else:
            return self.name

    def invoke(self, arg_tuple):
        ''' Invoke the function with the given argument tuple '''
        # - rewrite. Should perform the following:
        # - assign its input tuple to the paramsTuple (which writes to its memory)
        # - perform the evaluation

        if len(arg_tuple) > len(self.params):
            raise EvaluationError(f"Function '{self.name}' expects {len(self.params)} parameters but received {len(arg_tuple)}")
        self.expression.parsed = self.expression.parsed_pos = None
        closure = self.closure.copy()
        self.params.assign(arg_tuple, closure)
        # evaluate the expression
        return self.expression.value(mem=closure)

    def __mul__(self, other):
        if not isinstance(other, Function):
            raise EvaluationError('Incorrect type for function composition')
        return FuncComposition(*self.func_list, *other.func_list)

    def __pow__(self, other):
        other = int(other)
        if other <= 0:
            raise EvaluationError('Functional power must be a positive integer')
        return FuncComposition(*(self.func_list * int(other)))

class FuncComposition(Function):
    ''' Function composition class representing the composition of multiple functions '''
    def __init__(self, *func_list):
        self.name = ''.join([fn.name for fn in func_list])
        self.func_list = list(func_list)
        self.function = self.invoke

    def __str__(self):
        return self.name

    def invoke(self, arg_tuple):
        res = arg_tuple
        for fn in self.func_list[::-1]:
            res.tokens[0] = fn.invoke(res)
        return res.tokens[0]

class LFunc(Function, LValue):
    ''' LValue Function class representing a function definition without invocation '''
    def __init__(self, wordtoken, params):
        self.name = wordtoken.name
        self.params = params

    def __str__(self):
        return f"{self.name}{self.params}"
