''' Module op '''

from calc_operators import Operator, Prefix, Postfix, Infix, Ternary, PrefixFunction
from calc_functions import Function
from calc_errors import CalculatorError, EvaluationError
from calc_vars import LValue
from calc_number import RealNumber, ComplexNumber, Number, zero, one, two, three, four, half, ten, ln2, ln10, ln1_1, sqrt2, sqrt_2pi, pi, imag_i, onePointOne, e
from calc_settings import Settings

st = Settings()

def factorial_fn(n):
    ''' factorial function n!'''
    if not isinstance(n, RealNumber):
        raise EvaluationError("Invalid argument to factorial operator")
    if not n.is_int():
        raise CalculatorError(f'Factorial operator expects an integer, not {str(n)}')
    n = int(n)
    if n in (0, 1):
        return one
    for i in range(2, n):
        n *= i
    return RealNumber(n)

def permutation_fn(n, r):  # nPr
    ''' permutation function '''
    return combination_fn(n, r, perm=True)

def combination_fn(n, r, perm=False):  # nCr
    ''' combination function '''
    if not isinstance(n, RealNumber) or not isinstance(r, RealNumber):
        raise EvaluationError("Invalid argument to combination operator")
    if not n.is_int() or not r.is_int():
        raise EvaluationError('Combination function expects integers')
    n, r = int(n), int(r)
    res = 1
    if n in (0, 1):
        return one
    for i in range(1, r + 1):
        res *= n + 1 - i
        res //= i ** (not perm)
    return RealNumber(res)

def exponentiation_fn(a, b):
    ''' exponentiation function a^b '''
    from calc_tuples import Tuple
    if isinstance(a, Tuple) or isinstance(b, Tuple):
        raise EvaluationError("Cannot perform exponentiation with tuples/vectors")
    if isinstance(a, Function):
        if b.is_int():
            return a ** b
        raise CalculatorError(f'Cannot raise a function ({str(a)}) to a fractional power {str(b)}')
    if a == zero:
        if b == zero:
            raise CalculatorError('0^0 is undefined')
        return zero
    if b.is_int() and (isinstance(a, RealNumber) or isinstance(a, ComplexNumber) and a.real.is_int() and a.im.is_int()):
        return int_power(a, int(b))
    elif isinstance(a, ComplexNumber) and isinstance(b, RealNumber):  # complex ^ real
        r = exponentiation_fn(abs(a), b)
        theta = (a.arg() / pi).fast_continued_fraction() * b % two * pi
        return ComplexNumber(r * cos_fn(theta), r * sin_fn(theta)).fast_continued_fraction()
    # if isinstance(b, RealNumber) and b.sign == -1: return one / exponentiation_fn(a, -b, *args, fcf=fcf, **kwargs)
    # a^b = e^(b ln a)
    return exp(b * lnFn(a))

def int_power(base, power):
    ''' integer exponentiation function base^power '''
    if not isinstance(power, int):
        raise CalculatorError(f'int_power() expects integral power, received {power}')
    power = abs(power)
    result = one
    while power > 0:
        if power & 1:
            result *= base
        base = (base * base).fast_continued_fraction()
        power >>= 1
    return (one / result if power < 0 else result).fast_continued_fraction()

def exp(x):
    ''' exponential function e^x '''
    if isinstance(x, ComplexNumber):  # e^(a + ib) = (e^a) e^(ib) = (e^a) cis b
        r = exp(x.real)
        return ComplexNumber(r * cos_fn(x.im), r * sin_fn(x.im))
    intPart = int_power(e, int(x))
    x = x.frac_part()
    total = term = i = one
    while abs(term) >= st.epsilon:
        term = (term * x) / i
        total += term
        i += one
    return intPart * total.fast_continued_fraction()

def lnFn(x):
    ''' natural logarithm function ln(x) '''
    if not isinstance(x, Number):
        raise EvaluationError("Invalid argument to ln function")
    if x == zero:
        raise CalculatorError(f'ln 0 is undefined.')
    # ln(re^iθ) = ln r + iθ
    if isinstance(x, ComplexNumber):
        return ComplexNumber(lnFn(abs(x)), x.arg())
    if isinstance(x, RealNumber) and x < zero:
        return ComplexNumber(lnFn(abs(x)), pi)
    if x < one:
        return -lnFn(one / x)
    result = zero
    while x > ten:
        x /= ten
        result += ln10
    while x > two:
        x /= two
        result += ln2
    while True:
        x /= onePointOne
        result += ln1_1
        if x < one:
            break
    # ln (1 + x) = x - x^2/2 + x^3/3 - x^4/4 + x^5/5 - ...
    # ln (1 - x) = -x - x^2/2 - x^3/3 - x^4/4 - x^5/5 - ...
    x_pow = dx = x = one - x
    denom = one
    while abs(dx) > st.epsilon:
        result -= dx
        x_pow *= x
        denom += one
        dx = x_pow / denom 
    return result.fast_continued_fraction()

def sin_fn(x):
    ''' sine function sin(x) '''
    if isinstance(x, ComplexNumber):
        eiz = exp(imag_i * x)
        return (eiz - (one / eiz)) / two / imag_i
    elif not isinstance(x, Number):
        raise EvaluationError('Invalid argument to trig function')
    if x < zero:
        return -sin_fn(-x)
    x = x % (pi * two)
    if x > pi * three / two:
        return -sin_fn(pi * two - x)
    elif x > pi:
        return -sin_fn(x - pi)
    elif x > pi / two:
        return sin_fn(pi - x)
    total = x_pow = dx = x
    xSq = -x * x
    mul = fac = one
    while abs(dx) > st.epsilon:
        mul += two
        fac *= mul * (mul - one)
        x_pow *= xSq
        dx = (x_pow / fac).fast_continued_fraction()
        total += dx
    return total.fast_continued_fraction()

def cos_fn(x):
    ''' cosine function cos(x) '''
    return sin_fn(pi / two - x)

def tan_fn(x):
    ''' tangent function tan(x) '''
    return sin_fn(x) / cos_fn(x)

def sec_fn(x):
    ''' secant function sec(x) '''
    return one / cos_fn(x)

def csc_fn(x):
    ''' cosecant function csc(x) '''
    return one / sin_fn(x)

def cot_fn(x):
    ''' cotangent function cot(x) '''
    return one / tan_fn(x)

def sinh_fn(x):
    ''' hyperbolic sine function sinh(x) '''
    return ((ex := exp(x)) - one / ex) / two

def cosh_fn(x):
    ''' hyperbolic cosine function cosh(x) '''
    return ((ex := exp(x)) + one / ex) / two

def tanh_fn(x):
    ''' hyperbolic tangent function tanh(x) '''
    return ((e2x := exp(two * x)) - one) / (e2x + one)

def arcsin_fn(x):
    ''' inverse sine function arcsin(x) '''
    if not isinstance(x, RealNumber):
        raise EvaluationError("Invalid argument to arcsin function")
    # https://en.wikipedia.org/wiki/List_of_mathematical_series
    if x.sign == -1:
        return -arcsin_fn(-x)
    if x > one:
        raise CalculatorError('arcsin only accepts values from -1 to 1 inclusive')
    if x * x > -x * x + one:
        return pi / two - arcsin_fn(exponentiation_fn(-x * x + one, half))
    total = term = x
    xsqr = x * x
    k = zero
    while abs(term) > st.epsilon:
        k += one
        term *= xsqr * (k * two) * (k * two - one) / four / k / k
        term = term.fast_continued_fraction()
        total += term / (k * two + one)
    return total.fast_continued_fraction()

def arccos_fn(x):
    ''' inverse cosine function arccos(x) '''
    if not isinstance(x, RealNumber):
        raise EvaluationError("Invalid argument to arccos function")
    if x.sign == -1:
        return pi - arccos_fn(-x)
    if x > one:
        raise CalculatorError('arccos only accepts values from -1 to 1 inclusive')
    return pi / two - arcsin_fn(x)

def arctan_fn(x):
    ''' inverse tangent function arctan(x) '''
    if not isinstance(x, RealNumber):
        raise EvaluationError("Invalid argument to arctan function")
    if x.sign == -1:
        return -arctan_fn(-x)
    if x > one:
        return pi / two - arctan_fn(one / x)
    # https://en.wikipedia.org/wiki/Arctangent_series
    total = term = x / (x * x + one)
    factor = term * x
    num = two
    while abs(term) > st.epsilon:
        term *= factor
        term *= num
        term /= num + one
        term = term.fast_continued_fraction()
        total += term
        total = total.fast_continued_fraction()
        num += two
    return total

def abs_fn(x):
    ''' absolute value function abs(x) '''
    return abs(x)

def conj_fn(x): 
    ''' complex conjugate function conj(x) '''
    return x.conj()

def arg_fn(x): 
    ''' argument function arg(x) '''
    return x.arg()

def real_part_fn(x):
    ''' real part function Re(x) '''
    if isinstance(x, RealNumber):
        return x
    if isinstance(x, ComplexNumber):
        return x.real
    raise EvaluationError('Re() expects a complex number')

def im_part_fn(x):
    ''' imaginary part function Im(x) '''
    if isinstance(x, RealNumber):
        return zero
    if isinstance(x, ComplexNumber):
        return x.im
    raise EvaluationError('Im() expects a complex number')

def signum_fn(x):
    ''' signum function sgn(x) '''
    if not isinstance(x, RealNumber):
        raise EvaluationError('sgn() expects a real number')
    return one if x.sign == 1 else -one if x.sign == -1 else zero

def assignment_fn(L, R, mem=None):
    ''' assignment operator L = R '''
    from calc_tuples import LTuple
    if mem is None:
        raise MemoryError('No Memory object passed to assignment operator')
    if not isinstance(L, LValue):
        raise EvaluationError('Can only assign to LValue')
    if isinstance(L, LTuple):
        return L.assign(R, mem=mem)
    mem.add(L.name, R)
    return R

def index_fn(tup, idx):
    ''' indexing operator tup @ idx '''
    from calc_tuples import Tuple
    if not isinstance(tup, Tuple):
        raise EvaluationError("Index operator expects a tuple")
    if not isinstance(idx, RealNumber) or not idx.is_int():
        raise EvaluationError("Index must be an integer")
    idx = int(idx)
    if idx < 0 or idx >= len(tup):
        raise EvaluationError(f"index {idx} is out of bounds for this tuple")
    return tup.tokens[idx]

def tupLength_fn(tup):
    ''' tuple length operator tup$ '''
    from calc_tuples import Tuple
    if not isinstance(tup, Tuple):
        raise EvaluationError("Length operator expects a tuple")
    return RealNumber(len(tup))

def tup_concat_fn(tup1, tup2):
    ''' tuple concatenation operator tup1 <+> tup2 '''
    from calc_tuples import Tuple
    if not isinstance(tup1, Tuple) or not isinstance(tup2, Tuple):
        raise EvaluationError("Concatenation '<+>' expects tuples. End 1-tuples with ':)', e.g. '(3:)'")
    result = tup2.morphCopy()
    result.tokens = tup1.tokens + tup2.tokens
    return result

def knife_fn(dir):
    ''' knife operator dir is either '</' or '/>' '''
    def check(L, R):  # ensures that L is the index and R is the tuple.
        from calc_tuples import Tuple
        if not isinstance(L, RealNumber) or not isinstance(R, Tuple):
            return False
        if not L.is_int():
            raise EvaluationError("Knife operator expects an integer operand")
        L = int(L)
        if L < 0 or L > len(R):
            raise EvaluationError(f"Unable to slice {L} element(s) from this tuple")
        return True

    def knife(L, R):
        if check(L, R):
            tup = R.morphCopy()
            mid = int(L)
        elif check(R, L): # pylint: disable=W1114
            tup = L.morphCopy()
            mid = len(tup) - int(R)
        else: raise EvaluationError(f"Knife operator '{dir}' expects a tuple and an integer")
        tup.tokens = tup.tokens[mid:] if dir == '</' else tup.tokens[:mid]
        return tup

    return knife

def comparator(x, y):
    ''' comparator function: returns -1 if x<y, 0 if x==y, 1 if x>y '''
    from calc_tuples import Tuple
    match x, y:
        case Number(), Number(): return (x - y).fast_continued_fraction(epsilon=st.finalEpsilon)
        case Tuple(), Tuple():
            for i, j in zip(x, y):
                c = comparator(i, j)
                if c != zero: return c
            return zero if len(x) == len(y) else -one if len(x) < len(y) else one
        case _, _: raise EvaluationError("Unable to compare operands")

def lambda_arrow_fn(L, R, *args, **kwargs):
    ''' lambda arrow operator L => R '''
    # The function will already be created in R
    return R

def vector_dot_product_fn(L, R):
    ''' vector dot product operator L . R '''
    from calc_tuples import Tuple
    if not isinstance(L, Tuple) or not isinstance(R, Tuple):
        raise EvaluationError("Dot product '.' expects tuples/vectors")
    if len(L) != len(R) or len(L) == 0:
        raise EvaluationError("'.' expects non-empty tuples/vectors of the same length")
    result = None
    for x, y in zip(L.tokens, R.tokens):
        if result is None:
            result = x * y.conj()
        else: result += x * y.conj()
    return result

def vector_cross_product_fn(L, R):
    ''' vector cross product operator L >< R '''
    from calc_tuples import Tuple
    if not isinstance(L, Tuple) or not isinstance(R, Tuple):
        raise EvaluationError("Cross product '><' expects tuples/vectors")
    if len(L) != 3 or len(R) != 3:
        raise EvaluationError("'><' expects tuples/vectors of dimension 3")
    tup = L.morphCopy()
    tup.tokens = [L.tokens[1] * R.tokens[2] - R.tokens[1] * L.tokens[2], L.tokens[2] * R.tokens[0] - R.tokens[2] * L.tokens[0], L.tokens[0] * R.tokens[1] - R.tokens[0] * L.tokens[1]]
    return tup

def normal_pdf_fn(L):
    ''' normal probability density function normpdf(x, mu=0, sigma=1) '''
    from calc_tuples import Tuple
    if isinstance(L, Tuple):
        if len(L) != 3:
            raise EvaluationError('Expected normpdf(x) or normpdf(x, mu, sigma)')
        L = (L.tokens[0] - L.tokens[1]) / (sigma := L.tokens[2])
    else:
        sigma = one
    x_sqr = L * L
    return exp(-x_sqr / two) / sqrt_2pi / sigma

def normal_cdf_fn(L):
    ''' normal cumulative distribution function normcdf(a, b, mu=0, sigma=1) '''
    from calc_tuples import Tuple
    if isinstance(L, Tuple):
        if len(L) not in (2, 4):
            raise EvaluationError('Expected normcdf(a, b) or normcdf(a, b, mu, sigma)')
        if len(L) == 2:
            a, b, mu, sigma = *L.tokens, zero, one
        else:  # len(L) == 4:
            a, b, mu, sigma = L.tokens
    else:
        raise EvaluationError('Expected normcdf(a, b) or normcdf(a, b, mu, sigma)')
    from math import erf
    x1, x2 = (a - mu) / sigma, (b - mu) / sigma
    return RealNumber((erf(x2 / sqrt2) - erf(x1 / sqrt2)) / 2)

def inv_erf(x):  # returns a such that erf(a) = x
    ''' inverse error function '''
    from math import erf
    L, R = -4, 4
    for _ in range(60):
        M = (L + R) / 2
        if erf(M) > x: R = M
        elif erf(M) < x: L = M
        else: return M
    return (L + R) / 2


def inv_normal_cdf_fn(L):
    ''' inverse normal cumulative distribution function invnorm(x, mu=0, sigma=1) '''
    from calc_tuples import Tuple
    if isinstance(L, Tuple):
        if len(L) != 3:
            raise EvaluationError('Expected invnorm(x) or invnorm(x, mu, sigma)')
        L, mu, sigma = L.tokens
    else:
        mu, sigma = zero, one
    return mu + sigma * sqrt2 * RealNumber(inv_erf(two * L - one))

assignment = Infix(' = ', assignment_fn)
lambdaArrow = Infix(' => ', lambda_arrow_fn)
spaceSeparator = Infix(' ', lambda x, y: x * y)
semicolonSeparator = Infix('; ', lambda x, y: y)
ternary_if = Ternary(' ? ', lambda cond, true_val, false_val: true_val if cond else false_val)
ternary_else = Ternary(' : ')
permutation = Infix('P', permutation_fn)
combination = Infix('C', combination_fn)
ambiguousPlus = Operator('+?')
ambiguousMinus = Operator('-?')
addition = Infix(' + ', lambda x, y: x + y)
subtraction = Infix(' - ', lambda x, y: x - y)
multiplication = Infix(' * ', lambda x, y: x * y)
implicitMult = Infix('', lambda x, y: x * y)
implicitMultPrefix = Infix(' ', lambda x, y: x * y)
fracDiv = Infix('/', lambda x, y: x / y)
division = Infix(' / ', lambda x, y: x / y)
intDiv = Infix(' // ', lambda x, y: x // y)
modulo = Infix(' % ', lambda x, y: x % y)
positive = Prefix('+', lambda x: x)
negative = Prefix('-', lambda x: -x)
indexing = Infix(' @ ', index_fn)
leftKnife = Infix(' </ ', knife_fn('</'))
rightKnife = Infix(' /> ', knife_fn('/>'))
tupConcat = Infix(' <+> ', tup_concat_fn)
tupLength = Postfix('$', tupLength_fn)
lt = Infix(' < ', lambda x, y: one if comparator(x, y) < zero else zero)
ltEq = Infix(' <= ', lambda x, y: one if comparator(x, y) <= zero else zero)
gt = Infix(' > ', lambda x, y: one if comparator(x, y) > zero else zero)
gtEq = Infix(' >= ', lambda x, y: one if comparator(x, y) >= zero else zero)
eq = Infix(' == ', lambda x, y: one if comparator(x, y) == zero else zero)
neq = Infix(' != ', lambda x, y: one if comparator(x, y) != zero else zero)
ltAccurate = Infix(' <* ', lambda x, y: one if x < y else zero)
ltEqAccurate = Infix(' <== ', lambda x, y: one if x <= y else zero)
gtAccurate = Infix(' >* ', lambda x, y: one if x > y else zero)
gtEqAccurate = Infix(' >== ', lambda x, y: one if x >= y else zero)
eqAccurate = Infix(' === ', lambda x, y: one if x == y else zero)
neqAccurate = Infix(' !== ', lambda x, y: one if x != y else zero)
logicalAND = Infix(' && ', lambda x, y: x if x.fast_continued_fraction(epsilon=st.finalEpsilon) == zero else y)
logicalOR = Infix(' || ', lambda x, y: x if x.fast_continued_fraction(epsilon=st.finalEpsilon) != zero else y)
functionInvocation = Infix('<invoke>', lambda x, y, mem=None: x.invoke(y))
functionComposition = Infix('', lambda x, y: x.invoke(y))
sin = Prefix('sin', sin_fn)
cos = Prefix('cos', cos_fn)
tan = Prefix('tan', tan_fn)
sec = Prefix('sec', sec_fn)
csc = Prefix('csc', csc_fn)
cot = Prefix('cot', cot_fn)
sinh = Prefix('sinh', sinh_fn)
cosh = Prefix('cosh', cosh_fn)
tanh = Prefix('tanh', tanh_fn)
arcsin = Prefix('asin', arcsin_fn)
arccos = Prefix('acos', arccos_fn)
arctan = Prefix('atan', arctan_fn)
ln = Prefix('ln', lnFn)
lg = Prefix('lg', lambda x: lnFn(x) / ln10)
weakSin = Prefix('sin ', sin_fn)
weakCos = Prefix('cos ', cos_fn)
weakTan = Prefix('tan ', tan_fn)
weakSec = Prefix('sec ', sec_fn)
weakCsc = Prefix('csc ', csc_fn)
weakCot = Prefix('cot ', cot_fn)
weakSinh = Prefix('sinh ', sinh_fn)
weakCosh = Prefix('cosh ', cosh_fn)
weakTanh = Prefix('tanh ', tanh_fn)
weakArcsin = Prefix('asin ', arcsin_fn)
weakArccos = Prefix('acos ', arccos_fn)
weakArctan = Prefix('atan ', arctan_fn)
weakLn = Prefix('ln ', lnFn)
weakLg = Prefix('lg ', lambda x: lnFn(x) / ln10)
weakSqrt = Prefix('sqrt ', lambda x: exponentiation_fn(x, half))
sqrt = Prefix('sqrt', lambda x: exponentiation_fn(x, half))
signum = PrefixFunction('sgn', signum_fn)
absolute = PrefixFunction('abs', abs_fn)
argument = PrefixFunction('arg', arg_fn)
conjugate = PrefixFunction('conj', conj_fn)
realPart = PrefixFunction('Re', real_part_fn)
imPart = PrefixFunction('Im', im_part_fn)
exponentiation = Infix('^', exponentiation_fn)
factorial = Postfix('!', factorial_fn)
vectorDotProduct = Infix('.', vector_dot_product_fn)
vectorCrossProduct = Infix('><', vector_cross_product_fn)
normpdf = PrefixFunction('normpdf', normal_pdf_fn)
normcdf = PrefixFunction('normcdf', normal_cdf_fn)
invnorm = PrefixFunction('normcdf', inv_normal_cdf_fn)

regex = {
    r'\s*(<\/)\s*': leftKnife,
    r'\s*(\/>)\s*': rightKnife,
    r'\s*(\/\/)\s*': intDiv,
    r'\s+(\/)\s+': division,
    r'(?<!\s)(\/)(?!\s)': fracDiv,
    r'\s*(\*)\s*': multiplication,
    r'\s*(%)\s*': modulo,
    r'(!)': factorial,
    r'\s*(\^)\s*': exponentiation,
    r'\s*(\+)\s+': addition,
    r'\s*(\-)\s+': subtraction,
    r'\s*(><)\s*': vectorCrossProduct,
    r'\s*(>==)\s*': gtEqAccurate,
    r'\s*(<==)\s*': ltEqAccurate,
    r'\s*(===)\s*': eqAccurate,
    r'\s*(!==)\s*': neqAccurate,
    r'\s*(<\+>)\s*': tupConcat,
    r'\s*(>\*)\s*': gtAccurate,
    r'\s*(<\*)\s*': ltAccurate,
    r'\s*(>=)\s*': gtEq,
    r'\s*(<=)\s*': ltEq,
    r'\s*(==)\s*': eq,
    r'\s*(!=)\s*': neq,
    r'\s*(=>)\s*': lambdaArrow,
    r'\s*(>)\s*': gt,
    r'\s*(<)\s*': lt,
    r'\s*(\?)\s*': ternary_if,
    r'\s*(:)\s*': ternary_else,
    r'\s*(\+)': ambiguousPlus,
    r'\s*(\-)': ambiguousMinus,
    r'\s*(&&)\s*': logicalAND,
    r'\s*(\|\|)\s*': logicalOR,
    r'\s*(=)\s*': assignment,
    r'\s*(@)\s*': indexing,
    r'(\$)\s*': tupLength,
    r'(sinh)\s+': weakSinh,
    r'(cosh)\s+': weakCosh,
    r'(tanh)\s+': weakTanh,
    r'(sinh)(?![A-Za-z_])': sinh,
    r'(cosh)(?![A-Za-z_])': cosh,
    r'(tanh)(?![A-Za-z_])': tanh,
    r'(sin)\s+': weakSin,
    r'(cos)\s+': weakCos,
    r'(tan)\s+': weakTan,
    r'(sec)\s+': weakSec,
    r'(csc|cosec)\s+': weakCsc,
    r'(cot)\s+': weakCot,
    r'(arcsin|asin)\s+': weakArcsin,
    r'(arccos|acos)\s+': weakArccos,
    r'(arctan|atan)\s+': weakArctan,
    r'(sqrt)\s+': weakSqrt,
    r'(ln)\s+': weakLn,
    r'(lg)\s+': weakLg,
    r'(abs)(?![A-Za-z_])': absolute,
    r'(arg)(?![A-Za-z_])': argument,
    r'(conj)(?![A-Za-z_])': conjugate,
    r'(Re)(?![A-Za-z_])': realPart,
    r'(Im)(?![A-Za-z_])': imPart,
    r'(sgn)(?![A-Za-z_])': signum,
    r'(sin)(?![A-Za-z_])': sin,
    r'(cos)(?![A-Za-z_])': cos,
    r'(tan)(?![A-Za-z_])': tan,
    r'(sec)(?![A-Za-z_])': sec,
    r'(csc|cosec)(?![A-Za-z_])': csc,
    r'(cot)(?![A-Za-z_])': cot,
    r'(arcsin|asin)(?![A-Za-z_])': arcsin,
    r'(arccos|acos)(?![A-Za-z_])': arccos,
    r'(arctan|atan)(?![A-Za-z_])': arctan,
    r'(normpdf)(?![A-Za-z_])': normpdf,
    r'(normcdf)(?![A-Za-z_])': normcdf,
    r'(invnorm)(?![A-Za-z_])': invnorm,
    r'(sqrt)(?![A-Za-z_])': sqrt,
    r'(ln)(?![A-Za-z_])': ln,
    r'(lg)(?![A-Za-z_])': lg,
    r'\s*(\.)\s*': vectorDotProduct,
    r'(\s)\s*': spaceSeparator,
    r'\s*(;)\s*': semicolonSeparator,
    r'(P)': permutation,
    r'(C)': combination,
}

# (precedenceWhenEntering, precedenceAfterEntering)
# A recursive call is entered only when `precedenceWhenEntering > currentPrecedenceLevel`
# E.g.:
# - (11.1, 10.9)          -> Operator is right-associative, e.g. exponentiation 2^3^4
# - (10, 10) or (10, 9.8) -> Operator is left-associative, e.g. addition, multiplication

power = {
    functionInvocation: (10.1, 99),  # why did I choose such a low precedence for this?
    # functionInvocation: (13, 99),
    factorial: (12, 12.1),
    tupLength: (12, 12.1),
    implicitMult: (11, 11),
    exponentiation: (11.1, 10.9),
    sqrt: (11.1, 10.9),
    sin: (11.1, 10.9),
    cos: (11.1, 10.9),
    tan: (11.1, 10.9),
    sec: (11.1, 10.9),
    csc: (11.1, 10.9),
    cot: (11.1, 10.9),
    absolute: (11.1, 10.9),
    argument: (11.1, 10.9),
    conjugate: (11.1, 10.9),
    realPart: (11.1, 10.9),
    imPart: (11.1, 10.9),
    signum: (11.1, 10.9),
    sinh: (11.1, 10.9),
    cosh: (11.1, 10.9),
    tanh: (11.1, 10.9),
    arcsin: (11.1, 10.9),
    arccos: (11.1, 10.9),
    arctan: (11.1, 10.9),
    normpdf: (11.1, 10.9),
    normcdf: (11.1, 10.9),
    invnorm: (11.1, 10.9),
    ln: (11.1, 10.9),
    lg: (11.1, 10.9),
    negative: (11.1, 10.9),
    positive: (11.1, 10.9),
    indexing: (10, 10),
    leftKnife: (10, 10),
    rightKnife: (10, 10),
    implicitMultPrefix: (10, 10),
    fracDiv: (9.5, 9.5),
    weakSqrt: (11.1, 8.9),
    weakSin: (11.1, 8.9),
    weakCos: (11.1, 8.9),
    weakTan: (11.1, 8.9),
    weakSec: (11.1, 8.9),
    weakCsc: (11.1, 8.9),
    weakCot: (11.1, 8.9),
    weakSinh: (11.1, 8.9),
    weakCosh: (11.1, 8.9),
    weakTanh: (11.1, 8.9),
    weakArcsin: (11.1, 8.9),
    weakArccos: (11.1, 8.9),
    weakArctan: (11.1, 8.9),
    weakLn: (11.1, 8.9),
    weakLg: (11.1, 8.9),
    permutation: (9, 9),
    combination: (9, 9),
    intDiv: (8, 8),
    division: (8, 8),
    multiplication: (8, 8),
    modulo: (8, 8),
    spaceSeparator: (8, 8),
    vectorCrossProduct: (8, 8),
    vectorDotProduct: (8, 8),
    subtraction: (7, 7),
    addition: (7, 7),
    tupConcat: (7, 7),
    gtEq: (6, 6),
    gt: (6, 6),
    ltEq: (6, 6),
    lt: (6, 6),
    eq: (5, 5),
    neq: (5, 5),
    gtEqAccurate: (6, 6),
    gtAccurate: (6, 6),
    ltEqAccurate: (6, 6),
    ltAccurate: (6, 6),
    eqAccurate: (5, 5),
    neqAccurate: (5, 5),
    logicalAND: (4, 4),
    logicalOR: (3, 3),
    assignment: (2, 1.9),
    lambdaArrow: (2, 1.9),
    ternary_if: (2, 0.5),
    ternary_else: (0.1, 0.5),
    semicolonSeparator: (0.01, 0.01),
    # comma_separator: (1, 1),
    None: (0, 0),
}

# a ? b ? c : d : e
# a + (b=3) + 4^b(c=b+1)a!sinb + 7c
# 4  (b=4) yz^ab cos 7 b sin 3 + 8
# Postfix (UR) 10
# 9- Implicit Mult -8
# 9- Exponentiation -8
# 7- Prefix (UL) -7
# 6- Division / Multiplication -6
# 5- Addition / Subtraction -5
# 4-AND-4
# 3-OR-3
# 2-Assignment-2
# 1-Comma-1