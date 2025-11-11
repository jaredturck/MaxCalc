''' Numbers module '''

from re import match
from math import gcd
from calc_settings import Settings
from calc_errors import NumberError, CalculatorError, EvaluationError
from calc_vars import Value

st = Settings()

class Number(Value):
    ''' Defines a number '''

class RealNumber(Number):
    ''' Defines a real number '''

    @staticmethod
    def gcd(a, b):
        ''' Compute gcd of two integers '''
        while a != 0:
            a, b = b % a, a
        return b

    def __init__(self, *inp, fcf=True, epsilon=None, max_denom='inf'):
        if len(inp) == 1:
            inp = inp[0]
        if isinstance(inp, float):
            inp = str(inp)
        if isinstance(inp, int):
            self.sign = (inp > 0) - (inp < 0)
            self.numerator = inp * self.sign
            self.denominator = 1
        elif isinstance(inp, str):
            if m := match(r'^(-)?(\d+)(?:\.(\d*))?$', inp) or match(r'(-)?(\d*)\.(\d+)', inp):  # integer or float
                self.from_string = m.group(0)
                sign, integer, new_frac = ['' if x is None else x for x in m.groups()]
                self.numerator = int(integer + new_frac)
                self.denominator = 10 ** len(new_frac)
                self.sign = 0 if self.numerator == 0 else -1 if sign == '-' else 1
            elif m := match(r'^(-)?(\d+)\/(\d+)$', inp):  # fraction
                sign, num, denom = m.groups()
                self.numerator = int(num)
                self.denominator = int(denom)
                if self.denominator == 0:
                    raise ZeroDivisionError('Denominator cannot be 0')
                self.sign = 0 if self.numerator == 0 else -1 if sign == '-' else 1
            else:
                raise NumberError('Cannot parse string. Try "-43.642" or "243" or "-6/71" etc')
            # handle negative numbers
        elif isinstance(inp, tuple) and len(inp) == 2 and isinstance(inp[0], int) and isinstance(inp[1], int):
            self.sign = 0 if inp[0] == 0 else 1 if inp[0] > 0 else -1
            self.numerator, self.denominator = abs(inp[0]), abs(inp[1])
        else:
            raise NumberError("Usage: Number(int [, int] | float | str | (int, int), fcf=True, epsilon=None, max_denom=None)")

        if self.denominator != 1:
            self.simplify()
            if fcf:
                new_frac = self.fast_continued_fraction(epsilon=epsilon, max_denom=max_denom)
                self.numerator = new_frac.numerator
                self.denominator = new_frac.denominator

    def is_int(self):
        ''' Check if denominator is 1 '''
        return self.denominator == 1

    def __int__(self):
        return self.numerator // self.denominator * self.sign

    def __float__(self):
        return self.numerator / self.denominator * self.sign

    def dec(self, dp=25):
        ''' Return decimal representation up to dp decimal places '''
        s = '-' if self.sign == -1 else ''
        s += str(self.numerator // self.denominator)
        rem = self.numerator % self.denominator
        if rem == 0:
            return s
        rem_list = []
        s += '.'
        frac = ''
        while rem and len(frac) <= dp:
            if rem in rem_list:
                return s + frac[:(i := rem_list.index(rem))] + '(' + frac[i:] + ')*'  # repeated decimal representation
            rem_list.append(rem)
            rem *= 10
            frac = frac + str(rem // self.denominator)
            rem %= self.denominator

        if len(frac) <= dp or frac[-1] in '01234':
            return s + frac[:dp]
        # At this point we don't have a repeated decimal representation, but rounding is required
        s = [ch for ch in s + frac[:-1]]
        for i in range(len(s))[::-1]:
            match s[i]:
                case '-': s[i] = '-1'
                case '.': continue
                case '9': s[i] = '10' if i == 0 else '0'
                case x:
                    s[i] = str(int(x) + 1)
                    break
        return ''.join(s[:-1] if dp == 0 else s)

    def simplify(self):
        ''' Simplify the fraction '''
        div = gcd(self.numerator, self.denominator)
        self.numerator = self.numerator // div
        self.denominator = self.denominator // div
        return self

    def fast_continued_fraction(self, epsilon=None, max_denom='inf'):
        ''' Compute the best rational approximation using continued fractions '''
        if self.sign == -1:
            return -(-self).fast_continued_fraction(epsilon=epsilon, max_denom=max_denom)
        if epsilon is None:
            epsilon = st.epsilon

        lower, upper, alpha = (zero, one), (one, zero), abs(self)
        gamma = (alpha * lower[1] - lower[0]) / (-alpha * upper[1] + upper[0])
        prev = lower
        while True:
            s = gamma - gamma % one
            lower = (lower[0] + s * upper[0], lower[1] + s * upper[1])
            # print(lower, s, lower[0] / lower[1])
            if lower[1] != zero and abs(self - (num := lower[0] / lower[1])) < epsilon or gamma == s:
                return num
            if max_denom != 'inf' and lower[1] > max_denom:
                return RealNumber(prev, fcf=False)
            prev = lower
            lower, upper = upper, lower
            gamma = one / (gamma - s)

    def __abs__(self):
        return self if self.sign >= 0 else -self

    def arg(self):
        ''' Return pi for negative numbers, 0 for positive numbers '''
        if self.sign == 0:
            raise CalculatorError('arg(0) is undefined.')
        return zero if self.sign == 1 else pi

    def conj(self):
        ''' Return instance '''
        return self

    def __add__(self, other):
        # if isinstance(other, (int, float)): other = RealNumber(other)
        if not isinstance(other, RealNumber):
            return NotImplemented
        num = self.sign * self.numerator * other.denominator + other.sign * self.denominator * other.numerator
        denom = self.denominator * other.denominator
        return RealNumber(num, denom, fcf=False)

    def __neg__(self):
        return RealNumber(-self.sign * self.numerator, self.denominator, fcf=False)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if not isinstance(other, RealNumber):
            return NotImplemented
        return RealNumber(self.sign * other.sign * self.numerator * other.numerator, self.denominator * other.denominator, fcf=False)

    def __truediv__(self, other):
        # if isinstance(other, (int, float)): other = RealNumber(other)
        if not isinstance(other, RealNumber):
            return NotImplemented
        if other.sign == 0:
            raise ZeroDivisionError('Division by 0 (RealNumber)')
        return RealNumber(self.sign * other.sign * self.numerator * other.denominator, self.denominator * other.numerator, fcf=False)

    def __floordiv__(self, other):
        if not isinstance(other, RealNumber):
            raise CalculatorError("Cannot perform integer division on non-reals.")
        result = self / other
        result.numerator = result.numerator // result.denominator
        result.denominator = 1
        return result

    def __mod__(self, other):
        # if isinstance(other, (int, float)): other = RealNumber(other)
        if not isinstance(other, RealNumber):
            return NotImplemented
        if other.sign == 0:
            raise ZeroDivisionError('Cannot modulo by 0')
        int_pieces = self / other
        int_pieces = RealNumber(int_pieces.sign * int_pieces.numerator // int_pieces.denominator, fcf=False)
        return self - other * int_pieces

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            other = RealNumber(other)
        if not isinstance(other, RealNumber):
            raise NumberError('Expected another Number')
        return (self - other).sign == 1

    def __lt__(self, other):
        return -self > -other

    def __eq__(self, other):
        # if isinstance(other, (int, float)): other = RealNumber(other)
        if not isinstance(other, RealNumber):
            return NotImplemented
        return (self - other).sign == 0

    def __ne__(self, other):
        return not self == other

    def __ge__(self, other):
        return not self < other

    def __le__(self, other):
        return not self > other

    def frac_part(self):
        ''' Return the fractional part of the number '''
        return RealNumber(self.numerator % self.denominator * self.sign, self.denominator, fcf=False)

    def value(self, *args, **kwargs):
        return self

    def __str__(self):
        return ('-' if self.sign == -1 else '') + str(self.numerator) + ('' if self.denominator == 1 else '/' + str(self.denominator))

    def __repr__(self):
        return str(self)

    def disp(self, frac_max_length, decimal_places):
        ''' Display the number either as a fraction or decimal based on length '''
        if self.denominator == 1:
            return str(self)
        s = str(self)
        if len(s) <= frac_max_length:
            return s + ' = ' + self.dec(dp=decimal_places)
        return self.dec(dp=decimal_places)

    @staticmethod
    def from_scientific_notation(significand, exponent):  # params are strings
        ''' Create a RealNumber from scientific notation parts '''
        input_str = significand + 'E' + exponent
        num = RealNumber(significand, fcf=True, epsilon=st.epsilon)
        exponent = int(exponent)
        if exponent > 0:
            num *= RealNumber(10 ** exponent, fcf=False)
        elif exponent < 0:
            num /= RealNumber(10 ** (-exponent), fcf=False)
        num.from_string = input_str
        return num

# 'Interning' some useful constants
zero = RealNumber(0)
one = RealNumber(1)
onePointOne = RealNumber(11, 10, fcf=False)
two = RealNumber(2)
three = RealNumber(3)
four = RealNumber(4)
ten = RealNumber(10)
e = RealNumber('2.718281828459045235360287471353', fcf=False)
pi = RealNumber('3.1415926535897932384626433832795', fcf=False)
ln1_1 = RealNumber(167314056934657, 1755468904561492, fcf=False)
ln2 = RealNumber(1554903831458736, 2243252046704767, fcf=False)
ln10 = RealNumber(227480160645689, 98793378510888, fcf=False)
half = one / two
sqrt2 = RealNumber(12477253282759, 8822750406821, fcf=False)
sqrt_2pi = RealNumber(5017911669018, 2001857124091, fcf=False)


class ComplexNumber(Number):  # Must be non-real valued, i.e. must have an imaginary part.
    ''' Defines a complex number '''

    def __new__(cls, real, im=zero):
        if im == zero:
            return real
        return super().__new__(cls)

    def __init__(self, real, im):
        self.real = real
        self.im = im

    def dec(self, dp=25):
        ''' Return decimal representation up to dp decimal places '''
        if self.real.sign == 0:
            return f"{self.im.dec(dp) if abs(self.im) != one else ''}i"
        else:
            return f"{self.real.dec(dp)}{' + ' if self.im.sign == 1 else ' - '}{abs(self.im).dec(dp) if abs(self.im) != one else ''}i"

    def simplify(self):
        ''' Simplify both parts '''
        self.real.simplify()
        self.im.simplify()
        return self

    def fast_continued_fraction(self, epsilon=None, max_denom='inf'):
        ''' Compute the best rational approximation using continued fractions for both parts '''
        if epsilon is None:
            epsilon = st.epsilon
        return ComplexNumber(
            self.real.fast_continued_fraction(epsilon=epsilon, max_denom=max_denom),
            self.im.fast_continued_fraction(epsilon=epsilon, max_denom=max_denom)
        )

    def is_int(self):
        ''' Return False '''
        return False

    def __abs__(self):
        from calc_op import exponentiation_fn
        return exponentiation_fn(self.real * self.real + self.im * self.im, half)

    def arg(self):
        ''' Return argument of complex number '''
        if self.real == zero:
            return pi / two if self.im > zero else -pi / two
        from calc_op import arctan_fn
        argument = arctan_fn(self.im / self.real)
        if self.real < 0:
            argument += pi if self.im > zero else -pi
        # 2nd quad get -theta but supposed to be pi - theta
        # 3rd quad get theta but supposed to be -pi + theta
        return argument

    def conj(self):
        ''' Return complex conjugate '''
        return ComplexNumber(self.real, -self.im)

    def reciprocal(self):
        ''' Return reciprocal '''
        return self.conj() / (self.real * self.real + self.im * self.im)

    def __add__(self, other):
        if isinstance(other, RealNumber):
            return ComplexNumber(self.real + other, self.im)
        return ComplexNumber(self.real + other.real, self.im + other.im)

    def __radd__(self, other):
        if isinstance(other, RealNumber):
            return self + ComplexNumber(other, zero)
        return NotImplemented

    def __neg__(self):
        return ComplexNumber(-self.real, -self.im)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if other == zero:
            return zero
        if isinstance(other, RealNumber):
            return ComplexNumber(self.real * other, self.im * other)
        return ComplexNumber(self.real * other.real - self.im * other.im, self.real * other.im + self.im * other.real)

    def __rmul__(self, other):
        if isinstance(other, RealNumber):
            return self * other
        return NotImplemented

    def __truediv__(self, other):
        if other == zero:
            raise ZeroDivisionError('Division by 0 (ComplexNumber)')
        if isinstance(other, RealNumber):
            return ComplexNumber(self.real / other, self.im / other)
        if not isinstance(other, ComplexNumber):
            return NotImplemented
        return self * other.reciprocal()

    def __floordiv__(self, other):
        raise EvaluationError("Cannot perform integer division on non-reals.")

    def __rtruediv__(self, other):
        if isinstance(other, RealNumber):
            return self.reciprocal() * other
        return NotImplemented

    def __gt__(self, other):
        raise EvaluationError('Complex value has no total ordering')

    def __lt__(self, other):
        return -self > -other

    def __eq__(self, other):
        if isinstance(other, (int, float, RealNumber)):
            return False
        return self.real == other.real and self.im == other.im

    def __ne__(self, other):

        return not self == other
    def __ge__(self, other):

        return not self < other
    def __le__(self, other):
        return not self > other

    def value(self, *args, **kwargs):
        return self

    def __str__(self):
        if self.real.sign == 0:
            return f"{'-' if self.im.sign == -1 else ''}{str(abs(self.im)) if abs(self.im) != one else ''}{' ' if self.im.denominator != 1 else ''}i"
        else:
            return f"{str(self.real)}{' + ' if self.im.sign == 1 else ' - '}{str(abs(self.im)) if abs(self.im) != one else ''}{' ' if self.im.denominator != 1 else ''}i"

    def __repr__(self):
        return str(self)

    def disp(self, frac_max_length, decimal_places):
        if self.real.denominator == 1 and self.im.denominator == 1:
            return str(self)
        if len(str(self.real)) <= frac_max_length and len(str(self.im)) <= frac_max_length:
            return str(self) + ' = ' + self.dec(dp=decimal_places)
        return self.dec(dp=decimal_places)

imag_i = ComplexNumber(zero, one)

# test code
if __name__ == '__main__':
    st.epsilon = RealNumber(1, 10 ** 20, fcf=False)
    a = RealNumber(2, 5)
    b = RealNumber(1, 4)
    assert a + b == RealNumber(13, 20)
    assert a / b == RealNumber(8, 5)
    assert a * b == RealNumber(1, 10)
    assert a - b == RealNumber(3, 20)
    c = ComplexNumber(RealNumber(5), RealNumber(2))
    d = ComplexNumber(RealNumber(3), RealNumber(-1))
    c.arg()
